#!/usr/bin/env python3
"""Arm A7, the two-sided rate, on SMAP/MSL TUNE only -- docs/MODELS.md 82.2.

    PYTHONPATH=src .venv/bin/python scripts/d86_a7.py

A7 = max(z_err, |z_rate|): z_err is the flown smoothed-residual z (`zr`, as in A0);
|z_rate| is the two-sided z of the SIGNED first difference, its sd floored by 79.11's
rule for new terms. A0's rate term is the one-sided z of |dx|.

SMAP/MSL is read from R2 INTO MEMORY (d86_a6's reader, sha256-checked against the
manifest) and nothing but the result is written (no dataset copy on the Mac). The cached
univariate fits are read only; the store's manifest must not move (stop 61). TUNE only:
no EVAL channel's events are loaded. A0 must reproduce 79.4's cut and 13 TUNE catches
before any A7 number is reported. No wall-clock figure is printed (stop 35).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_eval import ops                                          # noqa: E402
from sentinel_models import registry                                    # noqa: E402

_spec = importlib.util.spec_from_file_location("d86_a6", ROOT / "scripts" / "d86_a6.py")
A6 = importlib.util.module_from_spec(_spec); sys.modules["d86_a6"] = A6
_spec.loader.exec_module(A6)
D = A6.D                                                                # d86_arms, as loaded

OUT = ROOT / "runs" / "d86" / "a7_tune.json"
ARMS7 = {"A0": ["zr", "zd"], "A7": ["zr", "zta"]}


def channel_terms(cid):
    train, test = D.smap_channel(cid)
    uni = registry.build("gru-telemanom"); uni.config = D.SR.proportional_config(len(test))
    v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
    D.load_cached(uni, v_tr, np.ones(len(v_tr), bool), D.SR.ctx_for(cid, len(v_tr)))
    cfg = uni.config
    x = v_te[:, 0].astype(np.float64)
    agg, _, _ = D.forecast_parts(uni, v_te)
    e_s = D.ewma(np.abs(x - agg[:, 0].astype(np.float64)), cfg.smoothing_window)
    ref = np.asarray(uni._smoothed_errors(v_te, D.SR.ctx_for(cid, len(v_te))), np.float64)[:, 0]
    if not np.array_equal(ref, e_s):
        raise RuntimeError("e_s differs from detectors._smoothed_errors")
    span = cfg.error_window
    scale = D.channel_scale(train[:, 0], unit=1.0)
    dx = np.diff(x, prepend=x[0])
    T = {"zr": D.zstat(e_s, span), "zd": D.zstat(D.derivative(x), span),
         "zta": np.abs(D.zstat_floored(dx, span, D.FLOOR_FRACTION * scale))}
    return dict(steps=len(x), warmup=int(uni.warmup_steps), T=T)


def build_per():
    spans = D.smap_labels()
    per, skipped = {}, {}
    for cid in D.smap_units():
        try:
            c = channel_terms(cid)
        except Exception as exc:
            skipped[cid] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
            continue
        anom = np.zeros(c["steps"], bool)
        for lo, hi in spans.get(cid, []):
            anom[lo:hi + 1] = True
        nominal = ~anom; nominal[:c["warmup"]] = False
        per[cid] = dict(steps=c["steps"], nominal=nominal, T=c["T"])
    return per, skipped


def main() -> int:
    before = D.store_manifest()
    client, budget, ledger, cfg = A6.load()
    target, _ = D.target_rate()
    per, skipped = build_per()
    tune = [t for t in D.smap_targets() if t[0] in D.TUNE]           # EVAL never loaded
    res = dict(section="docs/MODELS.md 82.2", target_rate=target, channels=len(per),
               skipped=skipped, tune_events=len(tune), arms={})
    for arm, names in ARMS7.items():
        sc = {c: np.maximum.reduce([per[c]["T"][n] for n in names]) for c in per}
        cut, rt = D.solve(per, sc, target)
        masks = {c: sc[c] >= cut for c in per}
        got = D.smap_caught(masks, tune, D.TUNE)
        att = {}
        for evn in got:
            c, lo = evn.split("["); lo = int(lo[:-1])
            hi = next(h for cc, ll, h in tune if cc == c and ll == lo)
            att[evn] = A6.attribution(per, names, cut, [(c, lo, hi)])
        res["arms"][arm] = dict(terms=names, cut=cut, rate=rt, tune=len(got), events=got,
                                attribution=att)
    a0 = res["arms"]["A0"]
    res["reproduction_A0"] = bool(abs(a0["cut"] - D.REPRO["A0"][0])
                                  <= D.CUT_RTOL * D.REPRO["A0"][0] + 5e-7
                                  and a0["tune"] == D.REPRO["A0"][1])
    s0, s7 = set(a0["events"]), set(res["arms"]["A7"]["events"])
    res["lost"], res["gained"] = sorted(s0 - s7), sorted(s7 - s0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"  channels {len(per)}, skipped {len(skipped)}, TUNE events {len(tune)}, "
          f"A0 reproduced: {res['reproduction_A0']}")
    if res["reproduction_A0"]:
        for arm in ARMS7:
            a = res["arms"][arm]
            print(f"    {arm}: cut {a['cut']:.6f}  rate {100 * a['rate']:.4f}%  TUNE {a['tune']}/{len(tune)}")
        print(f"  A7 loses {res['lost']}  gains {res['gained']}")
        for evn, at in res["arms"]["A7"]["attribution"].items():
            print(f"    {evn}: first {at.get('first')}  any {at.get('any')}")
    else:
        print("  (!) A0 did not reproduce: no A7 number is reported (MODELS 82.2)")
    ops.commit(client, cfg.bucket, ledger, budget)
    print(budget.report())
    if D.store_manifest() != before:
        print("  (!) STOP 61: the cached store moved"); return 61
    return 0


if __name__ == "__main__":
    sys.exit(main())
