#!/usr/bin/env python3
"""docs/MODELS.md 82.3: A0 against A7 on the testbed -- a true freeze and a legitimate pause.

    PYTHONPATH=src .venv/bin/python scripts/a7_powersim.py

Balanced plant, seed 1, retrain=0, 60,000 ticks, the flying model of 78.4. The event is
[30,000, 35,400), one orbit:
  twin    no event (the healthy twin)
  freeze  the LoadCurrent SENSOR sticks at its tick-29,999 value; the plant is untouched
  pause   the instrument is commanded off (plant mode pause=START,LEN), all channels consistent

A0 = max(z_res, z_der) as flown, cut 20.191072. A7 = max(z_res, |z_rate|), |z_rate| the
two-sided z of the signed first difference, floored by 79.11's rule. A7's cut is derived
by the toolkit's own rule for the flown cut (fit.py: q0.999 pooled settled nominal on the
fit half of the flying model's scored rows); A0 recalibrated on that path must reproduce
20.191072 first. Terms come from scripts/d86b.py's NumPy forecaster, which reproduces the
flight emits. Nothing in flight/ is touched. No wall-clock figure (stop 35).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_export.reader import read_model                          # noqa: E402
from sentinel_models import telemanom                                   # noqa: E402
from sentinel_toolkit.calibrate import calibrate                        # noqa: E402
from sentinel_toolkit.statistic import derivative, zstat, zstat_floored  # noqa: E402

_spec = importlib.util.spec_from_file_location("d86b", ROOT / "scripts" / "d86b.py")
B = importlib.util.module_from_spec(_spec); sys.modules["d86b"] = B
_spec.loader.exec_module(B)

LOOPSIM = ROOT / "oxcaml" / "_build" / "shape-c8-p10" / "loopsim"
FLYING = ROOT / "runs" / "d85" / "flying" / "flying.bin"
HEALTHY7 = ROOT / "runs" / "d85" / "flying" / "healthy_seed7.npy"
OUT = ROOT / "runs" / "a7_powersim"
SEED, TICKS, SPINUP, BALANCE = 1, 60_000, 60_000, "0.60,0.92"
EVENT, LEN, LOAD = 30_000, 5_400, 2
WINDOW, WARM = B.WINDOW, 2_350


def plant(name: str, extra: list[str]) -> np.ndarray:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(LOOPSIM), f"out={out}", f"flying={FLYING}", f"seed={SEED}",
                        f"ticks={TICKS}", f"spinup={SPINUP}", f"balance={BALANCE}", "retrain=0",
                        *extra], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    return np.fromfile(out / "trace.f32", np.float32).reshape(-1, 8).astype(np.float64)


def per_channel(w, spec, x, scale):
    t = B.terms(w, spec, x)                                 # zr, zd as flown, (T, 8)
    zta = np.stack([np.abs(zstat_floored(np.diff(x[:, c], prepend=x[0, c]), WINDOW,
                                         B.FLOOR_FRACTION * scale[c])) for c in range(8)], axis=1)
    return {"A0": np.maximum(t["zr"], t["zd"]).max(axis=1), "A7": np.maximum(t["zr"], zta).max(axis=1),
            "zr": t["zr"].max(axis=1), "zd": t["zd"].max(axis=1), "zta": zta.max(axis=1)}


def main() -> int:
    status, spec = read_model(FLYING.read_bytes())
    assert int(status) == 0
    w = B.weights_of(spec)
    seg = np.load(B.SEGMENT)["train"].astype(np.float64)
    scale = np.where(np.ptp(seg, axis=0) > 0, seg.std(axis=0), 1.0)

    # -- the cuts, by fit.py's rule on the flying model's own scored rows --------------
    h7 = np.load(HEALTHY7).astype(np.float64)
    scored = h7[h7.shape[0] // 2:]
    s = per_channel(w, spec, scored, scale)
    cuts = {}
    for arm in ("A0", "A7"):
        settled = s[arm][WARM:]
        half = len(settled) // 2
        cal = calibrate(settled[:half], settled[half:], span=WINDOW, quantile=0.999)
        cuts[arm] = dict(cut=float(cal.cut), fit_rate=float(cal.fit_rate),
                         holdout_rate=float(cal.holdout_rate))
    flown = float(spec["params"]["threshold"])
    gate = abs(cuts["A0"]["cut"] - flown) <= 1e-5 * flown
    res = {"section": "docs/MODELS.md 82.3", "flown_cut": flown, "cuts": cuts,
           "reproduction_A0": bool(gate)}
    if not gate:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "result.json").write_text(json.dumps(res, indent=1))
        print(f"  (!) A0 recalibrated to {cuts['A0']['cut']:.6f}, not {flown:.6f}: nothing reported")
        return 1

    # -- the three runs ---------------------------------------------------------------
    twin = plant("twin", [])
    freeze = twin.copy()
    freeze[EVENT:EVENT + LEN, LOAD] = twin[EVENT - 1, LOAD]
    pause = plant("pause", [f"pause={EVENT},{LEN}"])
    assert np.array_equal(pause[:EVENT], twin[:EVENT]), "the pause run differs before its start"
    runs = {"twin": twin, "freeze": freeze, "pause": pause}
    lo, hi = EVENT, EVENT + LEN + WINDOW
    res["runs"] = {}
    for name, x in runs.items():
        st = per_channel(w, spec, x, scale)
        row = {}
        for arm in ("A0", "A7"):
            alarm = st[arm] >= cuts[arm]["cut"]
            alarm[:WARM] = False
            after = np.flatnonzero(alarm[EVENT:]) + EVENT
            first = int(after[0]) if after.size else None
            terms = ["zr", "zd"] if arm == "A0" else ["zr", "zta"]
            row[arm] = {"first_alarm_at_or_after_event": first,
                        "first_minus_event": None if first is None else first - EVENT,
                        "first_terms": None if first is None else
                        [t for t in terms if st[t][first] >= cuts[arm]["cut"]],
                        "alarms_in_event_window": int(alarm[lo:hi].sum()),
                        "alarms_whole_run": int(alarm.sum())}
        res["runs"][name] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(f"  cuts: A0 {cuts['A0']['cut']:.6f} (flown {flown:.6f}, reproduced), "
          f"A7 {cuts['A7']['cut']:.6f}; held-out rates A0 {100 * cuts['A0']['holdout_rate']:.4f}% "
          f"A7 {100 * cuts['A7']['holdout_rate']:.4f}%")
    for name, row in res["runs"].items():
        for arm in ("A0", "A7"):
            r = row[arm]
            print(f"    {name:<6} {arm}: first alarm >= {EVENT}: {r['first_alarm_at_or_after_event']} "
                  f"(+{r['first_minus_event']}) via {r['first_terms']}; in [{lo}, {hi}): "
                  f"{r['alarms_in_event_window']}; whole run {r['alarms_whole_run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
