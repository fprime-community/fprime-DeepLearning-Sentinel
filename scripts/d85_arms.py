#!/usr/bin/env python3
"""D85 / docs/MODELS.md 78: the retraining loop's three arms on the testbed, as registered.

Every constant here is 78.4's and every rule is 78.5's or 78.6's; nothing is chosen
after a number is seen (stop 57). The runner links nothing itself: it drives
`loopsim` (the plant, the flight core, and the retrainer's RetrainLoop over the
generated cycle) and the toolkit's `gate`, and counts.

    PYTHONPATH=src .venv/bin/python scripts/d85_arms.py ladder
    PYTHONPATH=src .venv/bin/python scripts/d85_arms.py arm {a,b,c} --seed S --rung EMIS
    PYTHONPATH=src .venv/bin/python scripts/d85_arms.py controls

Outputs go under runs/d85/ (gitignored). Rates and leads are in TICKS; no timing figure
is produced (stop 35).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentinel_toolkit import gate as gate_mod  # noqa: E402

SHAPE = ROOT / "oxcaml" / "_build" / "shape-c8-p10"
LOOPSIM = SHAPE / "loopsim"
TOOLS = SHAPE / "ground_tools"
RUNS = ROOT / "runs" / "d85"
FLYING = RUNS / "flying" / "flying.bin"
CACHE = RUNS / "controls"

# -- 78.3 / 78.4, fixed before any detector run -----------------------------------------
SPINUP = 60_000
BALANCE = "0.60,0.92"
AGE_START, AGE_TAU, SOLAR_LOSS = 5_000, 8_000.0, 0.05
LADDER = (0.30, 0.45, 0.60)
SLOW_RATE, SLOW_START = 4.0e-5, 5_000
POST_RATE = 2.0e-4
GUARD, BUDGET, SCHEDULE, NOMINAL_PPM = 260, 1, 6_550, 1_046
HELD_LEN, WARM = gate_mod.HELD_LEN, gate_mod.WARM
BLOCK, BLOCKS, HORIZON = 5_400, 4, 100_000
A_FIRST_DATA_MIN = 23_421                 # ageing at 90%: 5,000 + 8,000 ln 10
PREMISE = (40_000, 60_000)                # 78.5's window
POST_FA = (WARM, WARM + 20_000)           # after the swap: [swap + 2,350, swap + 22,350)
SPAN = 250 + 10
#: 78.9: one length for every arm, fixed before any arm ran.
ARM_TICKS = 400_000

NAMES = ["SolarInput", "ChargeCurrent", "LoadCurrent", "BusVoltage", "CellTemp",
         "RadiatorTemp", "HeaterDuty", "StateOfCharge"]
YELLOW_LOW = [-2.0, -10.0, 0.5, 27.0, -5.0, -35.0, 0.0, 0.30]
YELLOW_HIGH = [300.0, 10.0, 9.5, 32.0, 40.0, 30.0, 1.0, 1.0]


def loopsim(out: Path, seed: int, ticks: int, *, retrain: bool, ageing: float | None = None,
            fault: tuple[int, float] | None = None, swap: tuple[int, Path] | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    args = [str(LOOPSIM), f"out={out}", f"flying={FLYING}", f"seed={seed}", f"ticks={ticks}",
            f"spinup={SPINUP}", f"balance={BALANCE}", f"guard={GUARD}", f"budget={BUDGET}",
            f"schedule={SCHEDULE}", f"nominal_ppm={NOMINAL_PPM}",
            f"retrain={1 if retrain else 0}"]
    if ageing is not None:
        args.append(f"ageing={AGE_START},{AGE_TAU},{SOLAR_LOSS},{ageing}")
    if fault is not None:
        args.append(f"fault={fault[0]},{fault[1]}")
    if swap is not None:
        args.append(f"swap={swap[0]},{swap[1]}")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"loopsim failed in {out}: {r.stderr.strip()}")
    return out


def emitted(run: Path) -> np.ndarray:
    return np.loadtxt(run / "ticks.csv", delimiter=",", skiprows=1, usecols=1, dtype=np.int64)


def column(run: Path, col: int) -> np.ndarray:
    return np.loadtxt(run / "ticks.csv", delimiter=",", skiprows=1, usecols=col, dtype=np.int64)


def first_yellow(run: Path) -> int:
    for line in (run / "summary.txt").read_text().splitlines():
        if line.startswith("first_yellow"):
            return int(line.split()[1])
    raise SystemExit(f"no first_yellow in {run}")


def telemetry(run: Path) -> np.ndarray:
    return np.fromfile(run / "trace.f32", dtype=np.float32).reshape(-1, 8)


def candidates(run: Path) -> list[dict]:
    with open(run / "candidates.csv") as f:
        return [{k: (int(v) if k != "file" else v) for k, v in row.items()}
                for row in csv.DictReader(f)]


def run_gate(run: Path, cand: dict) -> dict:
    tele = telemetry(run)
    res = gate_mod.gate(
        flying=FLYING, candidate=run / cand["file"], telemetry=tele, names=NAMES,
        first_data=cand["first_data_tick"], last_data=cand["last_data_tick"],
        windows=cand["steps"] // BUDGET, budget=BUDGET, tools=TOOLS,
        yellow_low=np.asarray(YELLOW_LOW), yellow_high=np.asarray(YELLOW_HIGH),
        block=BLOCK, blocks=BLOCKS, horizon=HORIZON, workdir=run / f"gate_{cand['index']}",
        cache=CACHE, uplink_dest="RetrainApproved.bin")
    report = run / f"gate_{cand['index']}.json"
    report.write_text(gate_mod.to_json(res))
    (run / f"gate_{cand['index']}.txt").write_text(gate_mod.render(res) + "\n")
    return {"index": cand["index"], "verdict": res.verdict, "reasons": res.reasons,
            "first": cand["first_data_tick"], "last": cand["last_data_tick"],
            "part_i": res.numbers.get("part_i_improvement"), "m": res.numbers.get("margin_m"),
            "F": res.numbers.get("floor_F"),
            "trend_inside": [t["channel"] for t in res.trend if t["inside_horizon"]],
            "report": str(report.relative_to(ROOT))}


def rate(em: np.ndarray, lo: int, hi: int) -> float:
    return float(em[lo:hi].sum()) / float(hi - lo)


def attributable(fault_em: np.ndarray, ctrl_em: np.ndarray, lo: int, hi: int) -> list[int]:
    """42.9.1's rule: a warning is the fault's only if the same-seed control lacks it."""
    n = min(len(fault_em), len(ctrl_em), hi)
    return [t for t in range(lo, n) if fault_em[t] == 1 and ctrl_em[t] == 0]


# -- 78.5: the ageing premise ------------------------------------------------------------
def ladder() -> dict:
    out = {"window": PREMISE, "rungs": []}
    for emis in LADDER:
        rung = {"emis_loss": emis, "seeds": []}
        for seed in (1, 2, 3):
            h = loopsim(RUNS / "ladder" / f"healthy_s{seed}", seed, PREMISE[1], retrain=False)
            a = loopsim(RUNS / "ladder" / f"aged_e{emis}_s{seed}", seed, PREMISE[1],
                        retrain=False, ageing=emis)
            hc = int(emitted(h)[PREMISE[0]:PREMISE[1]].sum())
            ac = int(emitted(a)[PREMISE[0]:PREMISE[1]].sum())
            rung["seeds"].append({"seed": seed, "healthy": hc, "aged": ac,
                                  "qualifies": bool(ac >= 2 * hc and ac > hc)})
        rung["qualifies"] = all(s["qualifies"] for s in rung["seeds"])
        out["rungs"].append(rung)
        if rung["qualifies"]:
            out["chosen"] = emis
            break
    out.setdefault("chosen", None)
    (RUNS / "ladder.json").write_text(json.dumps(out, indent=2))
    return out


# -- 78.6: the arms ----------------------------------------------------------------------
def arm_c(seed: int) -> dict:
    run = loopsim(RUNS / "c" / f"s{seed}", seed, ARM_TICKS, retrain=True)
    n = len(emitted(run))
    gates = [run_gate(run, c) for c in candidates(run) if c["last_data_tick"] + 1 + HELD_LEN <= n]
    res = {"arm": "c", "seed": seed, "gates": gates,
           "certified": sum(g["verdict"] == "CERTIFY" for g in gates)}
    (run / "result.json").write_text(json.dumps(res, indent=2))
    return res


def arm_b(seed: int) -> dict:
    run = loopsim(RUNS / "b" / f"s{seed}", seed, ARM_TICKS, retrain=True,
                  fault=(SLOW_START, SLOW_RATE))
    fy = first_yellow(run)
    ctrl = loopsim(RUNS / "b" / f"s{seed}_ctrl", seed, ARM_TICKS, retrain=False)
    n = len(emitted(run))
    end = fy if fy >= 0 else n          # 78.9: no yellow crossing -> gated to the end
    gates = [run_gate(run, c) for c in candidates(run)
             if c["last_data_tick"] + 1 + HELD_LEN <= n and c["last_data_tick"] >= SLOW_START
             and c["first_data_tick"] < end]
    attr = attributable(emitted(run), emitted(ctrl), SLOW_START, end)
    admitted = column(run, 5)
    ends = [t - GUARD - 1 for t in range(n) if admitted[t] == 1]
    after = sum(1 for e in ends if e >= SLOW_START)
    res = {"arm": "b", "seed": seed, "first_yellow": fy, "gates": gates,
           "certified": sum(g["verdict"] == "CERTIFY" for g in gates),
           "attributable_before_yellow": len(attr), "first_attributable": attr[0] if attr else None,
           "admitted_steps": len(ends), "admitted_after_onset": after,
           "share_after_onset": (after / len(ends)) if ends else None}
    (run / "result.json").write_text(json.dumps(res, indent=2))
    return res


def arm_a(seed: int, emis: float) -> dict:
    run = loopsim(RUNS / "a" / f"s{seed}", seed, ARM_TICKS, retrain=True, ageing=emis)
    n = len(emitted(run))
    pick = next((c for c in candidates(run) if c["first_data_tick"] >= A_FIRST_DATA_MIN
                 and c["last_data_tick"] + 1 + HELD_LEN <= n), None)
    res = {"arm": "a", "seed": seed, "emis_loss": emis}
    if pick is None:
        res["verdict"] = "NO CANDIDATE"
        (run / "result.json").write_text(json.dumps(res, indent=2))
        return res
    g = run_gate(run, pick)
    res["gate"] = g
    if g["verdict"] != "CERTIFY":
        (run / "result.json").write_text(json.dumps(res, indent=2))
        return res
    # The human's approval, represented by the one command's own checks.
    from sentinel_toolkit.approve import check
    check(json.loads((ROOT / g["report"]).read_text()), run / f"cand_{pick['index']}.bin")
    t_swap = pick["last_data_tick"] + 1 + HELD_LEN
    t_fault = t_swap + POST_FA[1]
    total = t_fault + 25_000
    swap = (t_swap, run / f"cand_{pick['index']}.bin")
    swapped = loopsim(RUNS / "a" / f"s{seed}_swap", seed, total, retrain=False, ageing=emis, swap=swap)
    aged = loopsim(RUNS / "a" / f"s{seed}_frozen", seed, total, retrain=False, ageing=emis)
    healthy = loopsim(RUNS / "a" / f"s{seed}_healthy", seed, total, retrain=False)
    faulted = loopsim(RUNS / "a" / f"s{seed}_swap_fault", seed, total, retrain=False, ageing=emis,
                      swap=swap, fault=(t_fault, POST_RATE))
    lo, hi = t_swap + POST_FA[0], t_swap + POST_FA[1]
    fa_sw, fa_aged, fa_h = rate(emitted(swapped), lo, hi), rate(emitted(aged), lo, hi), rate(emitted(healthy), lo, hi)
    recovery = ((fa_aged - fa_sw) / (fa_aged - fa_h)) if fa_aged > fa_h else None
    fy = first_yellow(faulted)
    attr = attributable(emitted(faulted), emitted(swapped), t_fault, fy if fy > 0 else total)
    res.update({"swap_tick": t_swap, "fa_window": [lo, hi], "fa_swapped": fa_sw,
                "fa_aged_frozen": fa_aged, "fa_healthy": fa_h, "recovery": recovery,
                "fault_tick": t_fault, "first_yellow": fy,
                "first_attributable": attr[0] if attr else None,
                "lead_ticks": (fy - attr[0]) if (attr and fy > 0) else None})
    (run / "result.json").write_text(json.dumps(res, indent=2))
    return res


def controls() -> None:
    """The six start-offset controls once, cached (they depend only on the flying model,
    the window count and the budget). A throwaway candidate is not needed: the gate's own
    control step is called with the flying file standing in for a candidate."""
    run = RUNS / "controls_prime"
    run.mkdir(parents=True, exist_ok=True)
    seg = gate_mod.segment_mod.read(FLYING, FLYING.read_bytes())
    rows = seg.train.shape[0]
    room = rows - (SCHEDULE + SPAN - 1)
    offsets = [k * room // (gate_mod.N_CONTROLS - 1) for k in range(gate_mod.N_CONTROLS)]
    seg_f32 = run / "segment.f32"
    np.ascontiguousarray(seg.train, dtype=np.float32).tofile(seg_f32)
    import hashlib
    key = hashlib.sha256(FLYING.read_bytes()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    # One at a time: the owner chose one core for D85's runs on the Mac (2026-09-26); the
    # remote box runs the arms in parallel, but the controls are cached before it starts.
    for o in offsets:
        out = CACHE / f"control_{key}_w{SCHEDULE}_b{BUDGET}_o{o}.bin"
        if not out.exists():
            subprocess.run([str(TOOLS), "train", str(FLYING), str(seg_f32), str(rows),
                            str(o), str(SCHEDULE), str(BUDGET), str(out)], check=True)
    print(f"controls cached: offsets {offsets}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ladder")
    sub.add_parser("controls")
    a = sub.add_parser("arm")
    a.add_argument("which", choices=("a", "b", "c"))
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--rung", type=float, default=None)
    args = ap.parse_args()
    if args.cmd == "ladder":
        print(json.dumps(ladder(), indent=2))
    elif args.cmd == "controls":
        controls()
    elif args.which == "c":
        print(json.dumps(arm_c(args.seed), indent=2))
    elif args.which == "b":
        print(json.dumps(arm_b(args.seed), indent=2))
    else:
        if args.rung is None:
            raise SystemExit("arm a needs --rung (the ladder's chosen emissivity loss)")
        print(json.dumps(arm_a(args.seed, args.rung), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
