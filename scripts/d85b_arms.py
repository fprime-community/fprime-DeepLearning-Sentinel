#!/usr/bin/env python3
"""D85b / docs/MODELS.md 81: the need threshold, arm (a)'s ladder, and the three arms.

Every constant here is 81's or 78.4's, and every rule is 81.2-81.5's, fixed before any
number was seen (stop 57). The runner links nothing itself. It drives `loopsim` (the
plant, the flight core, and the retrainer's RetrainLoop over the generated cycle) and the
toolkit's `gate`, and it counts. Every residual comes from `ground_tools residual`, the
flight core, via `gate.residual`.

    PYTHONPATH=src .venv/bin/python scripts/d85b_arms.py calibrate       81.2: T
    PYTHONPATH=src .venv/bin/python scripts/d85b_arms.py ladder          81.3: arm (a)'s L
    PYTHONPATH=src .venv/bin/python scripts/d85b_arms.py survey          81.9: the premise survey
    PYTHONPATH=src .venv/bin/python scripts/d85b_arms.py arm {a,b,c} --seed S

Outputs go under runs/d85b/ (gitignored). Rates and ticks only; no timing figure is
produced (stop 35).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentinel_toolkit import gate as gate_mod  # noqa: E402

SHAPE = ROOT / "oxcaml" / "_build" / "shape-c8-p10"
LOOPSIM = SHAPE / "loopsim"
TOOLS = SHAPE / "ground_tools"
RUNS = ROOT / "runs" / "d85b"
FLYING = ROOT / "runs" / "d85" / "flying" / "flying.bin"      # 78.4's; its sidecar beside it
CACHE = ROOT / "runs" / "d85" / "controls"                    # 78.4's six controls, cached

# -- 78.4, imported ----------------------------------------------------------------------
SPINUP, BALANCE = 60_000, "0.60,0.92"
GUARD, BUDGET, SCHEDULE, NOMINAL_PPM = 260, 1, 6_550, 1_046
BLOCK, BLOCKS, HORIZON = 5_400, 4, 100_000
WARM = gate_mod.WARM
# -- 81, fixed before any run ------------------------------------------------------------
ORBIT = 5_400
NEED_SPAN = gate_mod.WARM + gate_mod.NEED_ORBITS * ORBIT          # 23,950
HELD_SPAN = gate_mod.WARM + gate_mod.HELD_ORBITS * ORBIT          # 13,150
ARM_TICKS = 400_000
CAL_SEEDS = (1, 2, 3)                   # calibration: never an evaluation seed for arm (c)
C_SEEDS = (4, 5, 6)
AB_SEEDS = (1, 2, 3)
CAL_FIRST_END, CAL_STEP = 2 * NEED_SPAN, 1_000                    # windows end at 47,900 + 1,000 j
OCV_START, OCV_TAU = 30_000, 20_000.0
LADDER = (0.02, 0.04, 0.06)
LADDER_TICKS = 130_000
RATE_NORMAL_WINDOW = (30_000, 130_000)
RATE_NORMAL_TOL = 0.05
B_ONSET, B_RATE = 150_000, 4.0e-5
#: 81.9: the survey's family, in its registered order, each (loopsim key, magnitudes).
SURVEY = (("E", "eclipse", (-0.10, -0.20, -0.30)),
          ("H", "hk", (0.05, 0.10, 0.20)),
          ("D", "duty", (-0.10, -0.20, -0.30)),
          ("O", "ocv", (0.02, 0.04, 0.06)))

NAMES = ["SolarInput", "ChargeCurrent", "LoadCurrent", "BusVoltage", "CellTemp",
         "RadiatorTemp", "HeaterDuty", "StateOfCharge"]
YELLOW_LOW = np.array([-2.0, -10.0, 0.5, 27.0, -5.0, -35.0, 0.0, 0.30])
YELLOW_HIGH = np.array([300.0, 10.0, 9.5, 32.0, 40.0, 30.0, 1.0, 1.0])


# -- the rules, as functions a test can call ----------------------------------------------

def threshold(r_max: float) -> float:
    """81.2: T = 1 + max(1.5 x (R - 1), 0.02)."""
    return 1.0 + max(1.5 * (r_max - 1.0), 0.02)


def rate_normal(aged: np.ndarray, healthy: np.ndarray, lo: int, hi: int,
                tol: float = RATE_NORMAL_TOL) -> list[dict]:
    """81.3: per channel, |dx|'s 99.9th percentile and maximum over [lo, hi) within +/-tol
    (two-sided) of the same-seed healthy run's. A zero healthy statistic passes only if the
    aged one is zero too."""
    out = []
    for c, name in enumerate(NAMES):
        da = np.abs(np.diff(aged[lo - 1:hi, c].astype(np.float64)))
        dh = np.abs(np.diff(healthy[lo - 1:hi, c].astype(np.float64)))
        row = {"channel": name, "ok": True}
        for stat, fa, fh in (("p999", np.percentile(da, 99.9), np.percentile(dh, 99.9)),
                             ("max", da.max(), dh.max())):
            ratio = float(fa / fh) if fh > 0 else (1.0 if fa == 0 else float("inf"))
            row[stat] = ratio
            row["ok"] = bool(row["ok"] and (1.0 - tol) <= ratio <= (1.0 + tol))
        out.append(row)
    return out


def inside_yellow(tele: np.ndarray) -> bool:
    return bool(((tele >= YELLOW_LOW) & (tele <= YELLOW_HIGH)).all())


def gateable(cand: dict, n: int) -> bool:
    """81.2: its need window lies after the baseline, and both HELD windows fit the run."""
    end = cand["last_data_tick"] + 1
    return end >= 2 * NEED_SPAN and end + 2 * HELD_SPAN <= n


# -- apparatus ------------------------------------------------------------------------------

def scenario(key: str, delta: float) -> str:
    """81.3 / 81.9: one healthy change, on 81.3's schedule (start 30,000, tau 20,000)."""
    return f"{key}={OCV_START},{OCV_TAU},{delta}"


def loopsim(out: Path, seed: int, ticks: int, *, retrain: bool, ocv: float | None = None,
            fault: tuple[int, float] | None = None, swap: tuple[int, Path] | None = None,
            extra: str | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    args = [str(LOOPSIM), f"out={out}", f"flying={FLYING}", f"seed={seed}", f"ticks={ticks}",
            f"spinup={SPINUP}", f"balance={BALANCE}", f"guard={GUARD}", f"budget={BUDGET}",
            f"schedule={SCHEDULE}", f"nominal_ppm={NOMINAL_PPM}",
            f"retrain={1 if retrain else 0}"]
    if ocv is not None:
        args.append(scenario("ocv", ocv))
    if extra is not None:
        args.append(extra)
    if fault is not None:
        args.append(f"fault={fault[0]},{fault[1]}")
    if swap is not None:
        args.append(f"swap={swap[0]},{swap[1]}")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"loopsim failed in {out}: {r.stderr.strip()}")
    return out


def telemetry(run: Path) -> np.ndarray:
    return np.fromfile(run / "trace.f32", dtype=np.float32).reshape(-1, 8)


def res(model: Path, run: Path, lo: int, hi: int) -> float:
    rows = (run / "trace.f32").stat().st_size // 32
    return gate_mod.residual(TOOLS, model, run / "trace.f32", rows, lo, hi)


def baseline_of(model: Path, run: Path, start: int = 0) -> dict:
    return gate_mod.baseline(flying=model, telemetry=telemetry(run), start=start, orbit=ORBIT,
                             tools=TOOLS, workdir=run / "baseline")


def emitted(run: Path) -> np.ndarray:
    return np.loadtxt(run / "ticks.csv", delimiter=",", skiprows=1, usecols=1, dtype=np.int64)


def first_yellow(run: Path) -> int:
    for line in (run / "summary.txt").read_text().splitlines():
        if line.startswith("first_yellow"):
            return int(line.split()[1])
    raise SystemExit(f"no first_yellow in {run}")


def candidates(run: Path) -> list[dict]:
    with open(run / "candidates.csv") as f:
        return [{k: (int(v) if k != "file" else v) for k, v in row.items()}
                for row in csv.DictReader(f)]


# -- calibrate (81.2) ----------------------------------------------------------------------

def _calibrate_seed(seed: int) -> dict:
    run = loopsim(RUNS / "calibrate" / f"s{seed}", seed, ARM_TICKS, retrain=False)
    base = baseline_of(FLYING, run)
    rhos = []
    for e in range(CAL_FIRST_END, ARM_TICKS + 1, CAL_STEP):
        rhos.append((e, res(FLYING, run, e - NEED_SPAN, e) / base["residual"]))
    return {"seed": seed, "baseline": base, "rho": rhos}


def calibrate() -> None:
    with ProcessPoolExecutor(max_workers=len(CAL_SEEDS)) as pool:
        per_seed = list(pool.map(_calibrate_seed, CAL_SEEDS))
    r_max = max(r for s in per_seed for _, r in s["rho"])
    out = {"rule": "T = 1 + max(1.5 x (R - 1), 0.02); R = max rho over every window",
           "seeds": list(CAL_SEEDS), "windows_per_seed": len(per_seed[0]["rho"]),
           "R": r_max, "T": threshold(r_max),
           "per_seed": [{"seed": s["seed"], "baseline_residual": s["baseline"]["residual"],
                         "rho_min": min(r for _, r in s["rho"]),
                         "rho_max": max(r for _, r in s["rho"]),
                         "rho_max_at": max(s["rho"], key=lambda x: x[1])[0]} for s in per_seed]}
    (RUNS / "calibrate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("R", "T", "windows_per_seed")}, indent=2))
    for s in out["per_seed"]:
        print(f"  seed {s['seed']}: baseline {s['baseline_residual']:.6e}, rho "
              f"{s['rho_min']:.4f}..{s['rho_max']:.4f} (max at window ending {s['rho_max_at']:,})")


# -- ladder (81.3) --------------------------------------------------------------------------

def _ladder_unit(job: tuple[float, int]) -> dict:
    loss, seed = job
    run = loopsim(RUNS / "ladder" / f"L{loss}_s{seed}", seed, LADDER_TICKS, retrain=False, ocv=loss)
    aged = telemetry(run)
    healthy = telemetry(RUNS / "calibrate" / f"s{seed}")[:LADDER_TICKS]
    base = baseline_of(FLYING, run)
    rho = res(FLYING, run, LADDER_TICKS - NEED_SPAN, LADDER_TICKS) / base["residual"]
    rn = rate_normal(aged, healthy, *RATE_NORMAL_WINDOW)
    return {"loss": loss, "seed": seed, "rho": float(rho), "inside_yellow": inside_yellow(aged),
            "rate_normal": rn, "rate_normal_ok": all(r["ok"] for r in rn)}


def ladder() -> None:
    cal = json.loads((RUNS / "calibrate.json").read_text())
    T = cal["T"]
    jobs = [(loss, seed) for loss in LADDER for seed in AB_SEEDS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        units = list(pool.map(_ladder_unit, jobs))
    chosen, rungs = None, []
    for loss in LADDER:
        us = [u for u in units if u["loss"] == loss]
        ok = all(u["rho"] >= T and u["inside_yellow"] and u["rate_normal_ok"] for u in us)
        rungs.append({"loss": loss, "qualifies": ok, "units": us})
        if ok and chosen is None:
            chosen = loss
    out = {"T": T, "rule": "first rung with rho >= T, inside yellow and rate-normal on every seed",
           "rungs": rungs, "chosen": chosen}
    (RUNS / "ladder.json").write_text(json.dumps(out, indent=2))
    for r in rungs:
        for u in r["units"]:
            worst = max(r2 for row in u["rate_normal"] for r2 in (abs(row["p999"] - 1),
                                                                  abs(row["max"] - 1)))
            print(f"  L {u['loss']:.2f} seed {u['seed']}: rho {u['rho']:.4f} (T {T:.4f}), "
                  f"yellow {'ok' if u['inside_yellow'] else 'LEFT'}, rate-normal "
                  f"{'ok' if u['rate_normal_ok'] else 'FAIL'} (worst |ratio - 1| {worst:.4f})")
        print(f"  L {r['loss']:.2f}: {'QUALIFIES' if r['qualifies'] else 'does not qualify'}")
    print(f"  chosen: {chosen}")


# -- survey (81.9) --------------------------------------------------------------------------

def _survey_unit(job: tuple[str, str, float, int]) -> dict:
    member, key, delta, seed = job
    run = loopsim(RUNS / "survey" / f"{member}{delta}_s{seed}", seed, LADDER_TICKS,
                  retrain=False, extra=scenario(key, delta))
    aged = telemetry(run)
    healthy = telemetry(RUNS / "calibrate" / f"s{seed}")[:LADDER_TICKS]
    base = baseline_of(FLYING, run)
    rho = res(FLYING, run, LADDER_TICKS - NEED_SPAN, LADDER_TICKS) / base["residual"]
    rn = rate_normal(aged, healthy, *RATE_NORMAL_WINDOW)
    worst = max(rn, key=lambda r: max(abs(r["p999"] - 1), abs(r["max"] - 1)))
    return {"member": member, "key": key, "delta": delta, "seed": seed, "rho": float(rho),
            "inside_yellow": inside_yellow(aged), "rate_normal_ok": all(r["ok"] for r in rn),
            "worst_channel": worst["channel"], "worst_p999": worst["p999"],
            "worst_max": worst["max"]}


def choose(units: list[dict], T: float) -> dict | None:
    """81.9: the first member, in order, with a magnitude whose unit qualifies on every
    seed, at its smallest qualifying |delta|."""
    for member, key, deltas in SURVEY:
        for delta in sorted(deltas, key=abs):
            us = [u for u in units if u["member"] == member and u["delta"] == delta]
            if us and all(u["rho"] >= T and u["inside_yellow"] and u["rate_normal_ok"] for u in us):
                return {"member": member, "key": key, "delta": delta}
    return None


def survey() -> None:
    T = json.loads((RUNS / "calibrate.json").read_text())["T"]
    jobs = [(m, k, d, seed) for m, k, ds in SURVEY for d in ds for seed in AB_SEEDS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        units = list(pool.map(_survey_unit, jobs))
    chosen = choose(units, T)
    (RUNS / "survey.json").write_text(json.dumps({"T": T, "units": units, "chosen": chosen},
                                                 indent=2))
    for u in units:
        print(f"  {u['member']} {u['delta']:+.2f} s{u['seed']}: rho {u['rho']:.4f}, yellow "
              f"{'ok' if u['inside_yellow'] else 'LEFT'}, rate-normal "
              f"{'ok' if u['rate_normal_ok'] else 'FAIL'} ({u['worst_channel']} p99.9 "
              f"{u['worst_p999']:.3f}, max {u['worst_max']:.3f})")
    print(f"  chosen: {chosen}")


# -- the arms (81.4), run in session 2 -------------------------------------------------------

def _gate(run: Path, cand: dict, base: dict, T: float) -> dict:
    tele = telemetry(run)
    r = gate_mod.gate(
        flying=FLYING, candidate=run / cand["file"], telemetry=tele, names=NAMES,
        first_data=cand["first_data_tick"], last_data=cand["last_data_tick"],
        windows=cand["steps"] // BUDGET, budget=BUDGET, tools=TOOLS,
        yellow_low=YELLOW_LOW, yellow_high=YELLOW_HIGH, block=BLOCK, blocks=BLOCKS,
        horizon=HORIZON, workdir=run / f"gate_{cand['index']}", cache=CACHE,
        orbit=ORBIT, baseline_record=base, need_threshold=T, shadow=True)
    (run / f"gate_{cand['index']}.json").write_text(gate_mod.to_json(r))
    (run / f"gate_{cand['index']}.txt").write_text(gate_mod.render(r) + "\n")
    n = r.numbers
    return {"index": cand["index"], "first": cand["first_data_tick"],
            "last": cand["last_data_tick"], "verdict": r.verdict, "reasons": r.reasons,
            "need": n.get("need"), "shadow_verdict": n.get("shadow_verdict"),
            "windows": [{"held": w["held"], "F": w["floor_F"], "m": w["margin_m"],
                         "part_i": w["part_i_improvement"], "pass": not w["reasons"]}
                        for w in n.get("replication", [])]}


def arm(which: str, seed: int, loss: float | None) -> None:
    cal = json.loads((RUNS / "calibrate.json").read_text())
    T = cal["T"]
    out_dir = RUNS / which / f"s{seed}"
    kw = {}
    if which == "a":
        # 81.9: arm (a) is the survey's choice, read from its record, never chosen here.
        chosen = json.loads((RUNS / "survey.json").read_text())["chosen"]
        if chosen is None:
            raise SystemExit("81.9: the survey chose nothing; arm (a) is not run")
        kw["extra"] = scenario(chosen["key"], chosen["delta"])
        loss = chosen["delta"]
    if which == "b":
        kw["fault"] = (B_ONSET, B_RATE)
    run = loopsim(out_dir, seed, ARM_TICKS, retrain=True, **kw)
    base = baseline_of(FLYING, run)
    n = telemetry(run).shape[0]
    cands = candidates(run)
    result = {"arm": which, "seed": seed, "T": T, "loss": loss, "baseline": base,
              "candidates": len(cands), "gates": []}
    for cand in cands:
        if not gateable(cand, n):
            continue
        g = _gate(run, cand, base, T)
        result["gates"].append(g)
        if which == "a" and g["verdict"] == "CERTIFY":
            break                                   # 81.4: until the first CERTIFY
    result["certified"] = sum(1 for g in result["gates"] if g["verdict"] == "CERTIFY")
    result["need_triggers"] = sum(1 for g in result["gates"] if g["need"] and g["need"]["triggered"])
    if which == "b":
        end = first_yellow(run)
        end = n if end < 0 else end
        result["first_yellow"] = first_yellow(run)
        fail = [g for g in result["gates"] if g["last"] >= B_ONSET and g["first"] < end]
        result["failure_candidates"] = [g["index"] for g in fail]
        result["failure_certified"] = sum(1 for g in fail if g["verdict"] == "CERTIFY")
    if which == "a":
        cert = [g for g in result["gates"] if g["verdict"] == "CERTIFY"]
        if cert:
            g = cert[0]
            t_swap = g["last"] + 1 + 2 * HELD_SPAN
            post = (t_swap, t_swap + NEED_SPAN)
            cand_file = run / f"cand_{g['index']}.bin"
            c2 = {"t_swap": t_swap, "w_post": list(post)}
            if post[1] <= n:
                c2["res_candidate"] = res(cand_file, run, *post)
                c2["res_flying"] = res(FLYING, run, *post)
                c2["lower"] = bool(c2["res_candidate"] < c2["res_flying"])
                swapped = loopsim(out_dir.parent / f"s{seed}_swap", seed, post[1], retrain=False,
                                  swap=(t_swap, cand_file), extra=kw["extra"])
                em_u, em_s = emitted(run), emitted(swapped)
                lo, hi = post[0] + WARM, post[1]
                c2["fa_before"] = int(em_u[t_swap - 4 * ORBIT:t_swap].sum())
                c2["fa_after_unswapped"] = int(em_u[lo:hi].sum())
                c2["fa_after_swapped"] = int(em_s[lo:hi].sum())
                c2["fa_identical"] = bool((em_u[lo:hi] == em_s[lo:hi]).all())
            else:
                c2["lower"] = None
                c2["note"] = "W_post runs past the end of the run"
            result["c2"] = c2
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("gates", "baseline")}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("ladder")
    sub.add_parser("survey")
    a = sub.add_parser("arm")
    a.add_argument("which", choices=["a", "b", "c"])
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--loss", type=float, default=None)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.cmd == "calibrate":
        calibrate()
    elif args.cmd == "ladder":
        ladder()
    elif args.cmd == "survey":
        survey()
    else:
        seeds = C_SEEDS if args.which == "c" else AB_SEEDS
        if args.seed not in seeds:
            raise SystemExit(f"arm ({args.which}) runs at seeds {seeds} (81.4)")
        arm(args.which, args.seed, args.loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
