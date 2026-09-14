"""Per-timestep traces for two named events, for figures 1 and 2.

**It reproduces `scripts/decision_layer_arms.py`'s arms rather than inventing
them**, so a trace either shows what D65's artifact says it should or the
transcription is wrong. The two rules, exactly as that script builds them:

    frozen (stage 4)   telemanom.channel_ratios(e_s, cfg) >= 0.5505986
                       `decision_layer_arms.py:245-248`, multiplier from the
                       stage-4 artifact
    adopted (arm 2)    max(z_residual, z_derivative) >= 5.288128
                       `:224`, `:233-235`; the cut is `detail.arm2_nodis
                       .threshold` in the arms artifact, not re-solved here

**(!) The span is the study's, not flight's.** The arms study sets
`det.config = proportional_config(len(test))` (D47), so the trailing window the
z-scores are taken over is `0.05 * n`, not the flight component's compile-time
2,100. A trace drawn at 2,100 would not be the run the artifact records.

**Cost.** The event spans come from the visibility artifact already on disk, so
no labels table is fetched:

    smoke   1 ledger + 1 manifest                      = 2 Class B
    run     1 ledger + 1 manifest + train + test       = 4 Class B
                                                    + 1 Class A each, the ledger
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                      # noqa: E402
from sentinel_eval import ops                              # noqa: E402
from sentinel_models import telemanom                      # noqa: E402
from sentinel_toolkit.statistic import zstat               # noqa: E402


def _load(name, rel):
    # `sys.modules[name] = mod` before `exec_module` is load-bearing, not tidiness:
    # dataclasses resolve a field's type through `sys.modules[cls.__module__]`, and
    # without the registration that lookup returns None and the import dies inside
    # `dataclasses._is_type`. Same four lines as `decision_layer_arms.py:35-38`.
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ARMS = ROOT / "runs/smap-msl/_forensics/2026-09-10T182234Z-arms2.json"
VIS = ROOT / "runs/smap-msl/_forensics/2026-09-03T192723Z-visibility.json"
OUT = ROOT / "runs/smap-msl/_traces"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="E-10")
    ap.add_argument("--smoke", action="store_true", help="manifest only, fetch no array")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args(argv)

    arms = json.loads(ARMS.read_text())
    vis = json.loads(VIS.read_text())
    frozen_mult = float(arms["frozen_multiplier"])
    arm2_cut = float(arms["detail"]["arm2_nodis"]["threshold"])
    events = [(s["start"], s["end"]) for s in vis["sequences"]
              if s["channel"] == a.channel and s["class"] == "contextual" and s["in_range"]]
    frozen_caught = {e.split("[")[0] + "[" + e.split("[")[1] for e in arms["detail"]["frozen_eval"]}
    arm2_caught = set(arms["detail"]["arm2_nodis"]["eval"])
    print(f"  arms artifact: frozen_mult {frozen_mult:.6f}, arm2 cut {arm2_cut:.6f}")
    print(f"  {a.channel} in-range contextual events: {events}")

    SR = _load("smap_rungs", "scripts/smap_rungs.py")
    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    print(f"  month so far: {ledger.month_class_a} Class A, {ledger.month_class_b} Class B")

    projected = 2 if a.smoke else 4
    print(f"  projected {projected} Class B "
          f"(1 ledger + 1 manifest{'' if a.smoke else ' + train + test'})")
    man = json.loads(SR.fetch(client, cfg.bucket, SR.MANIFEST_KEY))
    ch = next(c for c in man["channels"] if c["channel_id"] == a.channel)

    if a.smoke:
        print(f"  actual {budget.class_b} Class B, projected {projected}: "
              f"{'MATCH' if budget.class_b == projected else '(!) DIVERGED'}")
        print(f"  {a.channel} objects: {[o['split'] for o in ch['objects']]} "
              f"-- the run will fetch {len(ch['objects'])}")
        if not a.no_ledger:
            ops.commit(client, cfg.bucket, ledger, budget)
        return 0 if budget.class_b == projected else 1

    keys = {o["split"]: o["key"] for o in ch["objects"]}
    train = np.load(io.BytesIO(SR.fetch(client, cfg.bucket, keys["train"])))
    test = np.load(io.BytesIO(SR.fetch(client, cfg.bucket, keys["test"])))
    actual_b = budget.class_b
    print(f"  actual {actual_b} Class B, projected {projected}: "
          f"{'MATCH' if actual_b == projected else '(!) DIVERGED'}")
    if actual_b != projected:
        print("  STOPPING: the projection was wrong.")
        if not a.no_ledger:
            ops.commit(client, cfg.bucket, ledger, budget)
        return 1

    store = SR.D.WEIGHT_STORE
    before = sum(1 for q in store.iterdir() if q.suffix == ".npz")

    det = SR.registry.build("gru-telemanom")
    det.config = SR.proportional_config(len(test))
    v_tr = train[:, :1].astype(np.float32)
    v_te = test[:, :1].astype(np.float32)
    det.fit(v_tr, np.ones(len(v_tr), dtype=bool), SR.ctx_for(a.channel, len(v_tr)))
    s_ctx = SR.ctx_for(a.channel, len(v_te))
    e_s = np.asarray(det._smoothed_errors(v_te, s_ctx), dtype=np.float64)[:, 0]
    forecast = np.asarray(det._forecast(det._filled(v_te), 0, len(v_te)),
                          dtype=np.float64)[:, 0]

    after = sum(1 for q in store.iterdir() if q.suffix == ".npz")
    assert after == before, f"weight store moved {before} -> {after}; this run must not fit"
    print(f"  weight store {before} -> {after} (cached weights, no refit)")

    x = v_te[:, 0].astype(np.float64)
    span = det.config.error_window
    z_res = zstat(e_s, span)
    z_der = zstat(np.abs(np.diff(x, prepend=x[0])), span)
    fused = np.maximum(z_res, z_der)
    ratios = np.asarray(telemanom.channel_ratios(e_s, det.config), dtype=np.float64)
    frozen_flag = ratios >= frozen_mult
    arm2_flag = fused >= arm2_cut
    print(f"  span {span} (proportional, D47), {len(x):,} test steps")

    # (!) The gate. A trace that disagrees with the artifact is a wrong
    # transcription, and the figure is not drawn from it.
    ok = True
    for lo, hi in events:
        tag = f"{a.channel}[{lo}]"
        fz_here = bool(frozen_flag[lo:hi + 1].any())
        a2_here = bool(arm2_flag[lo:hi + 1].any())
        fz_says, a2_says = tag in frozen_caught, tag in arm2_caught
        mark = "OK " if (fz_here == fz_says and a2_here == a2_says) else "(!)"
        ok &= (fz_here == fz_says and a2_here == a2_says)
        print(f"  {mark} {tag:<14} frozen {fz_here!s:<5} (artifact {fz_says!s:<5})   "
              f"adopted {a2_here!s:<5} (artifact {a2_says!s:<5})")
    if not ok:
        print("  STOPPING: the trace disagrees with the artifact. Not drawing a figure "
              "from it, and not substituting a different event.")
        if not a.no_ledger:
            ops.commit(client, cfg.bucket, ledger, budget)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.channel}-trace.npz"
    np.savez_compressed(
        path, actual=x, forecast=forecast, e_s=e_s, z_residual=z_res,
        z_derivative=z_der, fused=fused, frozen_ratio=ratios,
        frozen_flag=frozen_flag, adopted_flag=arm2_flag,
        meta=json.dumps({
            "channel": a.channel, "events": events, "span": int(span),
            "frozen_multiplier": frozen_mult, "arm2_cut": arm2_cut,
            "steps": int(len(x)), "source_arms": str(ARMS.relative_to(ROOT)),
            "source_visibility": str(VIS.relative_to(ROOT)),
            "producer": "scripts/dump_event_trace.py",
            "operations": budget.as_dict(),
        }))
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} B)")
    if not a.no_ledger:
        ops.commit(client, cfg.bucket, ledger, budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
