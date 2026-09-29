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


def gate(*, flying: Path, candidate: Path, telemetry: np.ndarray, names: list[str],
         first_data: int, last_data: int, windows: int, budget: int, tools: Path,
         yellow_low: np.ndarray, yellow_high: np.ndarray, block: int, blocks: int,
         horizon: int, workdir: Path, cache: Path | None = None,
         uplink_dest: str = "RetrainApproved.bin", dictionary: str = "<dictionary.json>",
         reload_command: str = "SentinelRef.sentinelMonitor.RELOAD_MODEL") -> GateResult:
    """Run the whole gate. `telemetry` holds every tick up to and past the candidate's
    held-out window; HELD is the HELD_LEN ticks after its last training tick."""
    res = GateResult(verdict="REFUSE")
    fb, cbytes = flying.read_bytes(), candidate.read_bytes()
    seg = segment_mod.read(flying, fb)

    # 1. integrity
    if not candidate_is_flying_with_new_weights(fb, cbytes):
        res.reasons.append("INTEGRITY: the candidate is not the flying file with new weights")
        return res
    held_lo = last_data + 1
    held_hi = held_lo + HELD_LEN
    if held_hi > telemetry.shape[0]:
        raise ToolkitError(f"HELD needs ticks {held_lo:,}..{held_hi:,}; the telemetry "
                           f"holds {telemetry.shape[0]:,}. Wait for the downlink.")
    workdir.mkdir(parents=True, exist_ok=True)
    tele_f32 = workdir / "telemetry.f32"
    np.ascontiguousarray(telemetry, dtype=np.float32).tofile(tele_f32)
    seg_f32 = workdir / "segment.f32"
    np.ascontiguousarray(seg.train, dtype=np.float32).tofile(seg_f32)
    rows_t, rows_s = telemetry.shape[0], seg.train.shape[0]

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

    r_c = [residual(tools, c, tele_f32, rows_t, held_lo, held_hi) for c in controls]
    r_s = residual(tools, candidate, tele_f32, rows_t, held_lo, held_hi)
    r_f = residual(tools, flying, tele_f32, rows_t, held_lo, held_hi)
    mean_c = float(np.mean(r_c))
    F = (max(r_c) - min(r_c)) / mean_c
    m = K * F
    part_i = (r_c[0] - r_s) / r_c[0]

    # 4. part (ii): generalisation gaps, candidate against flying
    fit_lo = max(first_data - WARM, 0)
    s_fit = residual(tools, candidate, tele_f32, rows_t, fit_lo, last_data + 1,
                     warm=first_data - fit_lo)
    f_fit = residual(tools, flying, seg_f32, rows_s, 0, rows_s)
    s_gap = abs(r_s - s_fit) / s_fit
    f_gap = abs(r_f - f_fit) / f_fit

    # 5. trend to limit, every channel
    trend = trend_table(telemetry, names, yellow_low, yellow_high, last_data + 1, block,
                        blocks, horizon)

    res.trend = trend
    res.numbers = {
        "flying_sha256": hashlib.sha256(fb).hexdigest(),
        "candidate_sha256": hashlib.sha256(cbytes).hexdigest(),
        "candidate_static_crc32": segment_mod.static_crc32(cbytes),
        "first_data_tick": first_data, "last_data_tick": last_data,
        "held": [held_lo, held_hi], "held_warm": WARM,
        "windows": windows, "budget": budget, "control_offsets": offsets,
        "res_held_controls": r_c, "res_held_candidate": r_s, "res_held_flying": r_f,
        "floor_F": F, "floor_bound": FLOOR_BOUND, "K": K, "margin_m": m,
        "part_i_improvement": part_i,
        "res_fit_candidate": s_fit, "res_fit_flying": f_fit,
        "gap_candidate": s_gap, "gap_flying": f_gap,
        "trend_block": block, "trend_blocks": blocks, "horizon": horizon}
    if F > FLOOR_BOUND:
        res.reasons.append(f"FLOOR: F = {F:.4%} exceeds {FLOOR_BOUND:.4%}; the floor is not a floor")
    if part_i < m:
        res.reasons.append(f"PART (i): improvement {part_i:.4%} < m = {m:.4%}")
    if s_gap > f_gap:
        res.reasons.append(f"PART (ii): candidate gap {s_gap:.4%} > flying gap {f_gap:.4%}")
    inside = [t["channel"] for t in trend if t["inside_horizon"]]
    if inside:
        res.reasons.append(f"TREND TO LIMIT: {', '.join(inside)} would reach a yellow limit "
                           f"within {horizon:,} ticks")
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
        lines.append("  REFUSED BECAUSE")
        lines += [f"    - {r}" for r in res.reasons]
    if n:
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
