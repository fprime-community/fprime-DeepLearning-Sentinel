#!/usr/bin/env python3
"""docs/MODELS.md 78.11: E1's tick ladder and LC2, read from the deployments' own logs.

LC2 asks whether the retrainer's replica emits on exactly the ticks the Monitor's detector
emits on. Neither emit stream is otherwise recorded: `CrossChannelWarning` is throttled at
10 and the Monitor's `Score` channel is not the fused score. So both deployments, started
with -E, write one line per emit, keyed by the tap's sequence number:

    ref.log      EMIT seq N          SentinelRef's EmitProbe, after the Monitor steps
                 TICK seq N          its heartbeat, every 1,000
                 PROBE_FIRST / PROBE_GAP / PROBE_STALE / PROBE_MODE
    retrain.log  REPLICA_EMIT seq N  the Retrainer, per sample fed to its replica
                 REPLICA_TICK seq N  its heartbeat, every 1,000
                 REPLICA_FIRST / REPLICA_GAP

(!) NOTHING HERE READS A TIMESTAMP. Both loggers stamp events with wall clock; stop 35
keeps every figure in ticks, so only sequence numbers are parsed.

    python scripts/d85_lc2.py lastseq RUN       the tap sequence both sides have reached
    python scripts/d85_lc2.py calibrate RUN     78.11's three rung conditions, PASS/FAIL
    python scripts/d85_lc2.py lc2 RUN           the verdict, and LoopSim as a third instance
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SEQ = re.compile(r"\b(EMIT|TICK|PROBE_FIRST|PROBE_STALE|PROBE_GAP|PROBE_MODE|REPLICA_EMIT|"
                 r"REPLICA_TICK|REPLICA_FIRST|REPLICA_GAP) seq (\d+)(?: model (\d))?")
SLIP = "RateGroupCycleSlip"
RELOAD = ("ModelReloadAccepted", "ModelReloadRolledBack", "ModelReloadWidthRefused")


@dataclass
class Side:
    """One deployment's log, reduced to sequence numbers."""
    emits: list[int] = field(default_factory=list)
    beats: list[int] = field(default_factory=list)
    first: int | None = None
    gaps: list[int] = field(default_factory=list)
    stale: list[int] = field(default_factory=list)
    modes: list[tuple[int, int]] = field(default_factory=list)
    slips: int = 0
    lost: int = 0
    reloads: int = 0
    degraded: int = 0

    @property
    def last(self) -> int:
        """The last heartbeat: every sequence up to it was processed on this side."""
        return max(self.beats) if self.beats else -1


def read_side(text: str, replica: bool) -> Side:
    s = Side()
    emit, beat = ("REPLICA_EMIT", "REPLICA_TICK") if replica else ("EMIT", "TICK")
    first = "REPLICA_FIRST" if replica else "PROBE_FIRST"
    gap = "REPLICA_GAP" if replica else "PROBE_GAP"
    for line in text.splitlines():
        if SLIP in line:
            s.slips += 1
        if "SamplesLost" in line:
            s.lost += 1
        if any(r in line for r in RELOAD):
            s.reloads += 1
        if "DegradedToBaseline" in line:
            s.degraded += 1
        m = SEQ.search(line)
        if not m:
            continue
        kind, seq = m.group(1), int(m.group(2))
        if kind == emit:
            s.emits.append(seq)
        elif kind == beat:
            s.beats.append(seq)
        elif kind == first and s.first is None:
            s.first = seq
        elif kind == gap:
            s.gaps.append(seq)
        elif kind == "PROBE_STALE":
            s.stale.append(seq)
        elif kind == "PROBE_MODE":
            s.modes.append((seq, int(m.group(3) or 0)))
    return s


def rg_slips(channel_log: str) -> int:
    """The largest `RgCycleSlips` value the ground received; 0 if none was ever sent.

    ActiveRateGroup writes the channel only when a slip moves it ("update on change"), so
    its absence is itself zero slips."""
    worst = 0
    for line in channel_log.splitlines():
        if "RgCycleSlips" in line:
            nums = re.findall(r"(-?\d+)\s*$", line.strip())
            if nums:
                worst = max(worst, int(nums[-1]))
    return worst


def calibrate(ref: Side, rep: Side, rg: int | None) -> list[tuple[str, bool, str]]:
    """78.11's three rung conditions.

    (!) `rg` is None when the GDS wrote no channel log. E1 found that this GDS
    configuration writes none, so the second condition read an absent file and passed
    vacuously on every rung. It is now reported as NOT MEASURED -- visibly, not as a
    pass -- and does not decide the rung: SentinelRef's own slips are what its
    `RateGroupCycleSlip` event in the first condition reports.
    """
    out = []
    out.append(("no RateGroupCycleSlip in either log", ref.slips == 0 and rep.slips == 0,
                f"ref {ref.slips}, retrain {rep.slips}"))
    if rg is None:
        out.append(("RgCycleSlips never above 0", True, "NOT MEASURED: no GDS channel log"))
    else:
        out.append(("RgCycleSlips never above 0", rg == 0, f"largest {rg}"))
    first_gap = rep.gaps[0] if rep.gaps else None
    out.append(("no sample lost, replica first sequence 0",
                rep.lost == 0 and not rep.gaps and rep.first == 0,
                f"SamplesLost {rep.lost}, first gap at {first_gap}, first sequence {rep.first}"))
    return out


def validity(ref: Side, rep: Side, rg: int | None) -> list[str]:
    """Every reason LC2 is NO VERDICT; empty when it may be read."""
    why = [f"{name}: {detail}" for name, ok, detail in calibrate(ref, rep, rg) if not ok]
    if ref.first != 0:
        why.append(f"the probe's first sequence is {ref.first}, not 0")
    if ref.gaps or ref.stale:
        why.append(f"the probe saw gaps {ref.gaps[:3]} or stale ticks {ref.stale[:3]}")
    if not ref.modes or ref.modes[0] != (0, 1) or len(ref.modes) != 1 or ref.degraded:
        why.append(f"the Monitor was not in MODEL mode throughout: {ref.modes[:3]}, "
                   f"DegradedToBaseline {ref.degraded}")
    if ref.reloads or rep.reloads:
        why.append(f"a reload happened ({ref.reloads + rep.reloads})")
    return why


def compare(a: list[int], b: list[int], hi: int) -> dict:
    """LC2's verdict over [0, hi]: equal sets, at least one emit each side."""
    sa = {x for x in a if 0 <= x <= hi}
    sb = {x for x in b if 0 <= x <= hi}
    only_a, only_b = sorted(sa - sb), sorted(sb - sa)
    if hi < 0:
        return {"verdict": "FAIL", "why": "no common range", "only_monitor": [], "only_replica": [],
                "n_monitor": 0, "n_replica": 0, "hi": hi}
    if not sa or not sb:
        verdict, why = "FAIL", "vacuous: a side has no emit in range"
    elif only_a or only_b:
        verdict, why = "FAIL", "the emit ticks differ"
    else:
        verdict, why = "HOLD", "identical"
    return {"verdict": verdict, "why": why, "only_monitor": only_a, "only_replica": only_b,
            "n_monitor": len(sa), "n_replica": len(sb), "hi": hi}


def loopsim_emits(ticks_csv: Path) -> tuple[list[int], int]:
    emits, last = [], -1
    with ticks_csv.open() as f:
        for row in csv.DictReader(f):
            t = int(row["tick"])
            last = t
            if row["emitted"] == "1":
                emits.append(t)
    return emits, last


def _sides(run: Path) -> tuple[Side, Side, int | None]:
    ref = read_side((run / "ref.log").read_text(errors="replace"), replica=False)
    rep = read_side((run / "retrain.log").read_text(errors="replace"), replica=True)
    chan = run / "gds" / "channel.log"
    rg = rg_slips(chan.read_text(errors="replace")) if chan.exists() else None
    return ref, rep, rg


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("lastseq", "calibrate", "lc2"):
        print(__doc__)
        return 2
    run = Path(argv[1])
    ref, rep, rg = _sides(run)
    if argv[0] == "lastseq":
        print(min(ref.last, rep.last))
        return 0
    if argv[0] == "calibrate":
        ok, measured = True, 0
        conds = calibrate(ref, rep, rg)
        for name, passed, detail in conds:
            ok = ok and passed
            unmeasured = detail.startswith("NOT MEASURED")
            measured += 0 if unmeasured else 1
            print(f"   {'----' if unmeasured else ('PASS' if passed else 'FAIL')}  {name}  ({detail})")
        print(f"   reached: probe {ref.last}, replica {rep.last}")
        if ok and measured < len(conds):
            print(f"   RUNG PASS on {measured} of {len(conds)} (the rest not measured)")
        else:
            print(f"   RUNG {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    why = validity(ref, rep, rg)
    hi = min(ref.last, rep.last)
    res = compare(ref.emits, rep.emits, hi)
    print(f"   range [0, {hi}]: Monitor {res['n_monitor']} emits, replica {res['n_replica']}")
    if why:
        print("   LC2: NO VERDICT")
        for w in why:
            print(f"     - {w}")
    else:
        print(f"   LC2: {res['verdict']} ({res['why']})")
    if res["only_monitor"] or res["only_replica"]:
        print(f"     Monitor only: {res['only_monitor'][:20]}")
        print(f"     replica only: {res['only_replica'][:20]}")
    ticks = run / "ground" / "ticks.csv"
    if ticks.exists():
        ls, ls_last = loopsim_emits(ticks)
        h = min(hi, ls_last)
        for name, stream in (("Monitor", ref.emits), ("replica", rep.emits)):
            third = compare(stream, ls, h)
            print(f"   LoopSim (third instance, not the verdict) against the {name} over "
                  f"[0, {h}]: {third['verdict']} ({third['why']}; "
                  f"{name} only {third['only_monitor'][:10]}, LoopSim only {third['only_replica'][:10]})")
    if why:
        return 3
    return 0 if res["verdict"] == "HOLD" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
