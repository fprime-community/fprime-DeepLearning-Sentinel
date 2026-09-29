"""D85: the ground gate -- a downlinked candidate judged against a REPRODUCED control.

`docs/DECISIONS.md` D85, with D78 and D74.4. A candidate the retrainer wrote is
certified only if ALL of these hold, and the report says which did not:

1. **Integrity.** The candidate is the flying file with new weights and nothing
   else changed (the bytes outside the weights and the two CRCs are identical),
   and the flying model's training segment is the one its sidecar declares.
2. **The floor.** N controls are reproduced ON THE GROUND with the same OxCaml cycle
   the retrainer flies (`ground_tools train`): warm from the flying weights, on the
   flying model's own stored segment, at the candidate's own window count and step
   budget, differing ONLY in the start offset into that segment (D85 decision 3 --
   the onboard trainer has no seed). F is their spread (range / mean of res_held),
   and it must not exceed 69.7's cold seed spread (31.9283%) or the floor is refused
   as unreliable.
3. **Part (i)** (D78): (res_held(C0) - res_held(S)) / res_held(C0) >= m = K * F,
   K = 2.0330 imported (stop 44), C0 the offset-0 control.
4. **Part (ii)** (D74.4, kept by D78 c.1): the candidate's generalisation gap
   |res_held - res_fit| / res_fit is no larger than the flying model's.
5. **Trend to limit** (D85 decision 4): for EVERY channel, over the trend window
   ending at the candidate's last training tick, the slope of the per-block maxima
   and minima, and the ticks until that slope would carry the channel to its yellow
   limit. Any channel inside the horizon H refuses. **The numbers are in the report
   for every channel, not only the verdict** -- the owner's ruling of 2026-09-25:
   the human who approves sees whether anything was heading for a limit.

res_held and res_fit are the flight core's own mean |x - forecast|
(`ground_tools residual`), so every residual is the one the detector would compute.

Nothing here transmits anything. The report carries the exact commands a human runs
to approve, and `sentinel_toolkit approve` is the one command that runs them.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import segment as segment_mod
from .errors import ToolkitError

#: D78 / 67.3, imported and not recomputed (stop 44).
K = 2.0330
#: 69.7's cold seed spread. A start-offset floor wider than this is not a floor.
FLOOR_BOUND = 0.319283
#: 69/76's geometry, imported (stop 45): the held-out window and its warm-up.
HELD_LEN = 9000
WARM = 2350
N_CONTROLS = 6
#: D85b / docs/MODELS.md 81.2. The need window and each HELD are WHOLE ORBITS after the
#: warm-up: 81.1 found every 78.10 certification set by where a 1.23-orbit HELD fell in the
#: orbit. The orbit is the mission's (`--orbit`); the counts are 81's.
NEED_ORBITS = 4
HELD_ORBITS = 2
REPLICATE = 2


@dataclass
class GateResult:
    verdict: str
    reasons: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    trend: list = field(default_factory=list)
    commands: list = field(default_factory=list)


def _le32(b: bytes, off: int) -> int:
    return int.from_bytes(b[off:off + 4], "little")


def candidate_is_flying_with_new_weights(flying: bytes, cand: bytes) -> bool:
    """MW4's check: only the weights payload and the two CRCs may differ."""
    if len(flying) != len(cand):
        return False
    cb, wb = _le32(flying, 32), _le32(flying, 36)
    lo, hi = 64 + cb, 64 + cb + wb
    for i in range(len(flying)):
        if flying[i] != cand[i]:
            if not (lo <= i < hi or 44 <= i < 48 or 60 <= i < 64):
                return False
    return True


def _run(args: list[str]) -> str:
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        raise ToolkitError(f"{Path(args[0]).name} {args[1]} failed: {out.stderr.strip()}")
    return out.stdout


def residual(tools: Path, model: Path, rows_file: Path, rows: int, lo: int, hi: int,
             warm: int = WARM) -> float:
    text = _run([str(tools), "residual", str(model), str(rows_file), str(rows), str(lo),
                 str(hi), str(warm)])
    return float(text.split("mean=")[1].split()[0])


def trend_table(values: np.ndarray, names: list[str], low: np.ndarray, high: np.ndarray,
                end: int, block: int, blocks: int, horizon: int) -> list[dict]:
    """Per channel: the slope of per-block maxima and minima over `blocks` blocks
    ending at `end`, and the ticks until that slope reaches the yellow limit."""
    start = end - block * blocks
    if start < 0:
        raise ToolkitError(f"the trend window needs {block * blocks:,} ticks before tick "
                           f"{end:,}; the telemetry holds {end:,}")
    x = np.arange(blocks, dtype=np.float64) * block + block / 2.0
    rows = []
    for c, name in enumerate(names):
        seg = values[start:end, c].astype(np.float64).reshape(blocks, block)
        mx, mn = seg.max(axis=1), seg.min(axis=1)
        s_hi = float(np.polyfit(x, mx, 1)[0])
        s_lo = float(np.polyfit(x, mn, 1)[0])
        to_hi = (float(high[c]) - float(mx[-1])) / s_hi if s_hi > 0 else float("inf")
        to_lo = (float(mn[-1]) - float(low[c])) / (-s_lo) if s_lo < 0 else float("inf")
        ticks = min(to_hi, to_lo)
        rows.append({
            "channel": name, "last_block_max": float(mx[-1]), "last_block_min": float(mn[-1]),
            "yellow_low": float(low[c]), "yellow_high": float(high[c]),
            "slope_of_maxima_per_tick": s_hi, "slope_of_minima_per_tick": s_lo,
            "ticks_to_yellow": ticks, "horizon": horizon,
            "margin_ticks": ticks - horizon, "inside_horizon": bool(ticks < horizon)})
    return rows


def need_span(orbit: int) -> int:
    """Ticks in a need or baseline window: the warm-up, then NEED_ORBITS whole orbits."""
    return WARM + NEED_ORBITS * orbit


def baseline(*, flying: Path, telemetry: np.ndarray, start: int, orbit: int, tools: Path,
             workdir: Path) -> dict:
    """D85b N1: the flying model's own residual over its first NEED_ORBITS whole orbits in
    flight, from `start` (the tick it began flying). Written once, when a model starts
    flying, and bound to that model by its sha256."""
    hi = start + need_span(orbit)
    if hi > telemetry.shape[0]:
        raise ToolkitError(f"the baseline needs ticks {start:,}..{hi:,}; the telemetry holds "
                           f"{telemetry.shape[0]:,}")
    workdir.mkdir(parents=True, exist_ok=True)
    tele = workdir / "baseline_telemetry.f32"
    np.ascontiguousarray(telemetry, dtype=np.float32).tofile(tele)
    r = residual(tools, flying, tele, telemetry.shape[0], start, hi)
    return {"flying_sha256": hashlib.sha256(flying.read_bytes()).hexdigest(),
            "window": [start, hi], "warm": WARM, "orbit": orbit,
            "need_orbits": NEED_ORBITS, "residual": r}


def gate(*, flying: Path, candidate: Path, telemetry: np.ndarray, names: list[str],
         first_data: int, last_data: int, windows: int, budget: int, tools: Path,
         yellow_low: np.ndarray, yellow_high: np.ndarray, block: int, blocks: int,
         horizon: int, workdir: Path, cache: Path | None = None,
         uplink_dest: str = "RetrainApproved.bin", dictionary: str = "<dictionary.json>",
         reload_command: str = "SentinelRef.sentinelMonitor.RELOAD_MODEL",
         orbit: int | None = None, baseline_record: dict | None = None,
         need_threshold: float | None = None, shadow: bool = False) -> GateResult:
    """Run the whole gate. `telemetry` holds every tick up to and past the candidate's
    held-out window(s).

    Without `orbit` this is D85's gate, unchanged: HELD is the HELD_LEN ticks after the
    candidate's last training tick.

    With `orbit` it is D85b's (docs/MODELS.md 81.2):
    - N1, the need: the flying model's residual over the NEED_ORBITS whole orbits ending at
      the last training tick, against `baseline_record`'s. Below `need_threshold` the verdict
      is NOT NEEDED and nothing is judged.
    - F1 and R1: REPLICATE back-to-back HELD windows of WARM + HELD_ORBITS orbits, and every
      one of D85's tests must pass on each.
    - `shadow` also judges a NOT NEEDED candidate, for the record only; it never certifies.
    """
    res = GateResult(verdict="REFUSE")
    fb, cbytes = flying.read_bytes(), candidate.read_bytes()
    seg = segment_mod.read(flying, fb)

    # 1. integrity
    if not candidate_is_flying_with_new_weights(fb, cbytes):
        res.reasons.append("INTEGRITY: the candidate is not the flying file with new weights")
        return res
    d85b = orbit is not None
    held_len = (WARM + HELD_ORBITS * orbit) if d85b else HELD_LEN
    n_windows = REPLICATE if d85b else 1
    helds = [(last_data + 1 + k * held_len, last_data + 1 + (k + 1) * held_len)
             for k in range(n_windows)]
    held_lo, held_hi = helds[0]
    if helds[-1][1] > telemetry.shape[0]:
        raise ToolkitError(f"HELD needs ticks {held_lo:,}..{helds[-1][1]:,}; the telemetry "
                           f"holds {telemetry.shape[0]:,}. Wait for the downlink.")
    workdir.mkdir(parents=True, exist_ok=True)
    tele_f32 = workdir / "telemetry.f32"
    np.ascontiguousarray(telemetry, dtype=np.float32).tofile(tele_f32)
    seg_f32 = workdir / "segment.f32"
    np.ascontiguousarray(seg.train, dtype=np.float32).tofile(seg_f32)
    rows_t, rows_s = telemetry.shape[0], seg.train.shape[0]

    # 1b. D85b N1: need first. Nothing is judged for a model that has not degraded.
    need = None
    if d85b:
        if baseline_record is None or need_threshold is None:
            raise ToolkitError("D85b needs the flying model's baseline and the need threshold")
        if baseline_record.get("flying_sha256") != hashlib.sha256(fb).hexdigest():
            raise ToolkitError("the baseline was measured for another flying model; a model's "
                               "baseline is written when it starts flying")
        if baseline_record.get("orbit") != orbit:
            raise ToolkitError("the baseline was measured at another orbit length")
        need_hi = last_data + 1
        need_lo = need_hi - need_span(orbit)
        if need_lo < baseline_record["window"][1]:
            raise ToolkitError(f"the need window {need_lo:,}..{need_hi:,} overlaps the "
                               f"baseline {baseline_record['window']}; too early to judge")
        r_need = residual(tools, flying, tele_f32, rows_t, need_lo, need_hi)
        rho = r_need / baseline_record["residual"]
        need = {"window": [need_lo, need_hi], "baseline_window": baseline_record["window"],
                "residual_need": r_need, "residual_baseline": baseline_record["residual"],
                "rho": rho, "threshold": need_threshold, "triggered": bool(rho >= need_threshold)}
        if not need["triggered"] and not shadow:
            res.verdict = "NOT NEEDED"
            res.reasons.append(f"NEED: rho = {rho:.4f} < T = {need_threshold:.4f}; the flying "
                               f"model has not degraded, so no candidate is judged")
            res.numbers = {"flying_sha256": hashlib.sha256(fb).hexdigest(),
                           "candidate_sha256": hashlib.sha256(cbytes).hexdigest(),
                           "first_data_tick": first_data, "last_data_tick": last_data,
                           "need": need, "orbit": orbit}
            return res

    # 2. the floor: N controls, differing only in start offset
    span = int(seg.meta["window"]) + int(seg.meta["n_predictions"])
    room = rows_s - (windows + span - 1)
    if room < N_CONTROLS - 1:
        raise ToolkitError(f"the stored segment ({rows_s:,} rows) cannot hold {N_CONTROLS} "
                           f"controls of {windows:,} windows of {span} rows")
    offsets = [k * room // (N_CONTROLS - 1) for k in range(N_CONTROLS)]
    cache_dir = cache or workdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(fb).hexdigest()[:16]
    controls, procs = [], []
    for o in offsets:
        out = cache_dir / f"control_{key}_w{windows}_b{budget}_o{o}.bin"
        controls.append(out)
        if not out.exists():
            procs.append(subprocess.Popen(
                [str(tools), "train", str(flying), str(seg_f32), str(rows_s), str(o),
                 str(windows), str(budget), str(out)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
    for p in procs:
        _, err = p.communicate()
        if p.returncode != 0:
            raise ToolkitError(f"a control failed to train: {err.strip()}")

    # 3. part (ii)'s fit residuals, once per candidate
    fit_lo = max(first_data - WARM, 0)
    s_fit = residual(tools, candidate, tele_f32, rows_t, fit_lo, last_data + 1,
                     warm=first_data - fit_lo)
    f_fit = residual(tools, flying, seg_f32, rows_s, 0, rows_s)

    # 4. every HELD window: the floor, part (i), part (ii)
    per_window = []
    for lo, hi in helds:
        r_c = [residual(tools, c, tele_f32, rows_t, lo, hi) for c in controls]
        r_s = residual(tools, candidate, tele_f32, rows_t, lo, hi)
        r_f = residual(tools, flying, tele_f32, rows_t, lo, hi)
        F = (max(r_c) - min(r_c)) / float(np.mean(r_c))
        m = K * F
        part_i = (r_c[0] - r_s) / r_c[0]
        s_gap = abs(r_s - s_fit) / s_fit
        f_gap = abs(r_f - f_fit) / f_fit
        reasons = []
        tag = f"HELD {lo:,}..{hi:,}: " if d85b else ""
        if F > FLOOR_BOUND:
            reasons.append(f"{tag}FLOOR: F = {F:.4%} exceeds {FLOOR_BOUND:.4%}; the floor is not a floor")
        if part_i < m:
            reasons.append(f"{tag}PART (i): improvement {part_i:.4%} < m = {m:.4%}")
        if s_gap > f_gap:
            reasons.append(f"{tag}PART (ii): candidate gap {s_gap:.4%} > flying gap {f_gap:.4%}")
        per_window.append({"held": [lo, hi], "res_held_controls": r_c, "res_held_candidate": r_s,
                           "res_held_flying": r_f, "floor_F": F, "margin_m": m,
                           "part_i_improvement": part_i, "gap_candidate": s_gap,
                           "gap_flying": f_gap, "reasons": reasons})

    # 5. trend to limit, every channel
    trend = trend_table(telemetry, names, yellow_low, yellow_high, last_data + 1, block,
                        blocks, horizon)

    first = per_window[0]
    res.trend = trend
    res.numbers = {
        "flying_sha256": hashlib.sha256(fb).hexdigest(),
        "candidate_sha256": hashlib.sha256(cbytes).hexdigest(),
        "candidate_static_crc32": segment_mod.static_crc32(cbytes),
        "first_data_tick": first_data, "last_data_tick": last_data,
        "held": first["held"], "held_warm": WARM,
        "windows": windows, "budget": budget, "control_offsets": offsets,
        "res_held_controls": first["res_held_controls"],
        "res_held_candidate": first["res_held_candidate"],
        "res_held_flying": first["res_held_flying"],
        "floor_F": first["floor_F"], "floor_bound": FLOOR_BOUND, "K": K,
        "margin_m": first["margin_m"], "part_i_improvement": first["part_i_improvement"],
        "res_fit_candidate": s_fit, "res_fit_flying": f_fit,
        "gap_candidate": first["gap_candidate"], "gap_flying": first["gap_flying"],
        "trend_block": block, "trend_blocks": blocks, "horizon": horizon}
    if d85b:
        res.numbers.update({"orbit": orbit, "need": need, "replication": per_window})
    for w in per_window:
        res.reasons += w["reasons"]
    inside = [t["channel"] for t in trend if t["inside_horizon"]]
    if inside:
        res.reasons.append(f"TREND TO LIMIT: {', '.join(inside)} would reach a yellow limit "
                           f"within {horizon:,} ticks")
    if need is not None and not need["triggered"]:
        # shadow: judged for the record, never certified
        res.numbers["shadow_verdict"] = "WOULD CERTIFY" if not res.reasons else "WOULD REFUSE"
        res.reasons.insert(0, f"NEED: rho = {need['rho']:.4f} < T = {need['threshold']:.4f}; "
                              f"judged in shadow only")
        res.verdict = "NOT NEEDED"
        return res
    if not res.reasons:
        res.verdict = "CERTIFY"
        # (!) A RELATIVE DESTINATION OF AT MOST 40 CHARACTERS: a command carries no more
        # (docs/MODELS.md 73.6), and fprime-cli's default is an absolute '/<name>'.
        if len(uplink_dest) > 40:
            raise ToolkitError(f"uplink destination {uplink_dest!r} is over 40 characters")
        res.commands = uplink_commands(candidate, uplink_dest, dictionary, reload_command)
        res.numbers["uplink_destination"] = uplink_dest
    return res


def uplink_commands(candidate, uplink_dest: str, dictionary, reload_command: str) -> list[str]:
    """The two lines a CERTIFY report names: the file uplink, then the reload.

    One function, so the report and anything that only PRINTS them (docs/MODELS.md 78.11:
    E1 under D85.1 shows the human's step and sends nothing) cannot drift apart.
    """
    return [
        f"fprime-cli file-uplink {candidate} {uplink_dest} --dictionary {dictionary}",
        f"fprime-cli command-send {reload_command} --arguments {uplink_dest} "
        f"--dictionary {dictionary}"]


def render(res: GateResult) -> str:
    n = res.numbers
    lines = ["=" * 76, f"  GROUND GATE: {res.verdict}", "=" * 76]
    if res.reasons:
        lines.append("  NOT JUDGED BECAUSE" if res.verdict == "NOT NEEDED" else "  REFUSED BECAUSE")
        lines += [f"    - {r}" for r in res.reasons]
    need = n.get("need") if n else None
    if need:
        lines += [
            "  THE NEED, FIRST (D85b N1: the flying model's own residual, whole orbits)",
            f"    need window {need['window'][0]:,}..{need['window'][1]:,}: "
            f"{need['residual_need']:.6e}; baseline {need['baseline_window'][0]:,}.."
            f"{need['baseline_window'][1]:,}: {need['residual_baseline']:.6e}",
            f"    rho {need['rho']:.4f} against T {need['threshold']:.4f}: "
            f"{'TRIGGERED' if need['triggered'] else 'not triggered'}"]
        if "shadow_verdict" in n:
            lines.append(f"    shadow (for the record, never decisive): {n['shadow_verdict']}")
    if n and "res_held_controls" in n and n.get("replication"):
        lines.append("  REPLICATION (D85b R1: every test on each whole-orbit HELD window)")
        for w in n["replication"]:
            lines.append(f"    HELD {w['held'][0]:,}..{w['held'][1]:,}: F {w['floor_F']:.4%}, "
                         f"m {w['margin_m']:.4%}, part (i) {w['part_i_improvement']:.4%}, "
                         f"gaps {w['gap_candidate']:.4%} / {w['gap_flying']:.4%}: "
                         f"{'pass' if not w['reasons'] else 'FAIL'}")
    if n and "res_held_controls" in n:
        lines += [
            "  THE CONTROL, REPRODUCED ON THE GROUND (D78, D85)",
            f"    controls (start offsets {n['control_offsets']}):",
            "      " + ", ".join(f"{r:.6e}" for r in n["res_held_controls"]),
            f"    floor F {n['floor_F']:.4%} (bound {n['floor_bound']:.4%}); "
            f"m = K x F = {n['K']} x F = {n['margin_m']:.4%}",
            f"    part (i)  candidate {n['res_held_candidate']:.6e} vs control "
            f"{n['res_held_controls'][0]:.6e}: improvement {n['part_i_improvement']:.4%}",
            f"    part (ii) gap candidate {n['gap_candidate']:.4%}, flying {n['gap_flying']:.4%}",
            f"    (flying model on HELD: {n['res_held_flying']:.6e})",
            f"    trained on ticks {n['first_data_tick']:,}..{n['last_data_tick']:,}; "
            f"HELD {n['held'][0]:,}..{n['held'][1]:,}; {n['windows']:,} windows x "
            f"{n['budget']} step(s)",
            "  TREND TO LIMIT, EVERY CHANNEL (the owner's ruling: the numbers, not a pass/fail)",
            f"    {'channel':<12}{'slope max/tick':>16}{'slope min/tick':>16}"
            f"{'ticks to yellow':>18}{'horizon':>10}{'margin':>16}"]
        for t in res.trend:
            tk = "never" if t["ticks_to_yellow"] == float("inf") else f"{t['ticks_to_yellow']:,.0f}"
            mg = "--" if t["ticks_to_yellow"] == float("inf") else f"{t['margin_ticks']:,.0f}"
            lines.append(f"    {t['channel']:<12}{t['slope_of_maxima_per_tick']:>16.3e}"
                         f"{t['slope_of_minima_per_tick']:>16.3e}{tk:>18}{t['horizon']:>10,}{mg:>16}"
                         + ("   <-- INSIDE" if t["inside_horizon"] else ""))
    if res.commands:
        lines += ["  IF A HUMAN APPROVES, ONE COMMAND:",
                  "    PYTHONPATH=src .venv/bin/python -m sentinel_toolkit approve --report <this report>.json",
                  "  which runs, and then verifies from the detector's own event log:"]
        lines += [f"    {c}" for c in res.commands]
    lines += ["  Status: the chosen retraining implementation, host-verified, not flight-qualified.",
              "=" * 76]
    return "\n".join(lines)


def to_json(res: GateResult) -> str:
    def clean(v):
        if isinstance(v, float) and v == float("inf"):
            return "inf"
        return v
    trend = [{k: clean(v) for k, v in t.items()} for t in res.trend]
    return json.dumps({"verdict": res.verdict, "reasons": res.reasons, "numbers": res.numbers,
                       "trend": trend, "commands": res.commands}, indent=2, sort_keys=True)
