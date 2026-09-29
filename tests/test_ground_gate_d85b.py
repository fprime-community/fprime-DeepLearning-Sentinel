"""D85b / docs/MODELS.md 81.2: the gate's need check, replication and whole-orbit windows,
each shown in both directions.

- N1: a flying model that has not degraded is NOT NEEDED -- nothing is judged, nothing is
  trained, and `approve` refuses the report. Degraded past T, the candidate is judged.
- R1: a candidate that passes the first whole-orbit HELD window and fails the second is
  REFUSED. Passing both certifies.
- F1: every window is the warm-up plus whole orbits, back to back.
- The shadow judges a NOT NEEDED candidate for the record and never certifies it.
- A baseline written for another model, or at another orbit, is refused; so is a need
  window that reaches back into the baseline.

The toolchain calls are stubbed, as in tests/test_ground_gate.py: the residual is a function
of the model and the window, so each rule can be driven directly.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sentinel_toolkit import approve as approve_mod
from sentinel_toolkit import gate as gate_mod
from sentinel_toolkit import segment as segment_mod
from sentinel_toolkit.errors import ToolkitError

ORBIT = 100                      # small, so the windows fit a small telemetry array
LAST = 30_000
NEED_SPAN = gate_mod.WARM + gate_mod.NEED_ORBITS * ORBIT
HELD_SPAN = gate_mod.WARM + gate_mod.HELD_ORBITS * ORBIT
GOOD_CONTROLS = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]      # F = 4.0%, m = 8.13%


def _model(n: int = 400, crc: int = 7) -> bytes:
    b = bytearray(n)
    b[32:36] = (40).to_bytes(4, "little")
    b[36:40] = (200).to_bytes(4, "little")
    b[44:48] = crc.to_bytes(4, "little")
    return bytes(b)


def _setup(tmp_path: Path):
    flying = tmp_path / "flying.bin"
    fb = _model()
    flying.write_bytes(fb)
    train = np.random.default_rng(0).normal(size=(7000, 2)).astype(np.float32)
    segment_mod.write(flying, fb, train, {"window": 250, "n_predictions": 10})
    cand = bytearray(fb)
    cand[64 + 40 + 5] ^= 0xFF
    cand[44] ^= 0x01
    candidate = tmp_path / "cand.bin"
    candidate.write_bytes(bytes(cand))
    return flying, candidate


class _Popen:
    calls = 0

    def __init__(self, args, **kw):
        _Popen.calls += 1
        Path(args[-1]).write_bytes(b"control")
        self.returncode = 0

    def communicate(self):
        return ("", "")


def _stub(monkeypatch, *, need: float, held_cand: list[float], held_flying: float = 1.0):
    """Baseline residual 1.0; the need window gives `need`; the candidate on HELD-k gives
    held_cand[k] and 0.80 on its own training span (so part (ii)'s gap is the HELD's); the
    controls give GOOD_CONTROLS on every window."""
    _Popen.calls = 0
    monkeypatch.setattr(subprocess, "Popen", _Popen)
    rows = 7000
    room = rows - (6550 + 260 - 1)
    offsets = [k * room // 5 for k in range(6)]
    held_starts = [LAST + 1 + k * HELD_SPAN for k in range(gate_mod.REPLICATE)]

    def fake(tools, model, rows_file, n, lo, hi, warm=gate_mod.WARM):
        name = Path(model).name
        if name.startswith("control_"):
            return GOOD_CONTROLS[offsets.index(int(name.split("_o")[1].split(".")[0]))]
        if rows_file.name == "segment.f32":
            return 1.0
        if name == "cand.bin":
            return held_cand[held_starts.index(lo)] if lo in held_starts else 0.80
        if lo == 0:
            return 1.0                                  # the baseline
        if lo == LAST + 1 - NEED_SPAN:
            return need                                 # the need window
        return held_flying
    monkeypatch.setattr(gate_mod, "residual", fake)


def _baseline(flying: Path, orbit: int = ORBIT, sha: str | None = None) -> dict:
    return {"flying_sha256": sha or hashlib.sha256(flying.read_bytes()).hexdigest(),
            "window": [0, NEED_SPAN], "warm": gate_mod.WARM, "orbit": orbit,
            "need_orbits": gate_mod.NEED_ORBITS, "residual": 1.0}


def _gate(tmp_path, flying, candidate, *, baseline=None, threshold=1.10, shadow=False,
          last=LAST):
    return gate_mod.gate(
        flying=flying, candidate=candidate, telemetry=np.zeros((60_000, 2), dtype=np.float32),
        names=["a", "b"], first_data=24_000, last_data=last, windows=6550, budget=1,
        tools=Path("ground_tools"), yellow_low=np.array([-10.0, -10.0]),
        yellow_high=np.array([10.0, 10.0]), block=5400, blocks=4, horizon=100_000,
        workdir=tmp_path / "w", cache=tmp_path / "c", orbit=ORBIT,
        baseline_record=baseline if baseline is not None else _baseline(flying),
        need_threshold=threshold, shadow=shadow)


# -- N1: need first --------------------------------------------------------------------

def test_a_model_that_has_not_degraded_is_not_judged(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.05, held_cand=[0.80, 0.80])
    res = _gate(tmp_path, flying, candidate)
    assert res.verdict == "NOT NEEDED" and not res.commands
    assert res.numbers["need"]["triggered"] is False
    assert _Popen.calls == 0, "a NOT NEEDED candidate trained controls"
    report = json.loads(gate_mod.to_json(res))
    with pytest.raises(ToolkitError, match="only a CERTIFY report"):
        approve_mod.check(report, candidate)


def test_a_degraded_model_lets_a_good_candidate_certify(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.20, held_cand=[0.80, 0.80])
    res = _gate(tmp_path, flying, candidate)
    assert res.verdict == "CERTIFY", res.reasons
    assert res.numbers["need"]["triggered"] and len(res.numbers["replication"]) == 2
    assert len(res.commands) == 2


def test_the_need_threshold_is_inclusive(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.10, held_cand=[0.80, 0.80])
    assert _gate(tmp_path, flying, candidate, threshold=1.10).verdict == "CERTIFY"


# -- R1: replication on two whole-orbit windows ----------------------------------------

def test_failing_the_second_window_refuses(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.20, held_cand=[0.80, 0.97])
    res = _gate(tmp_path, flying, candidate)
    assert res.verdict == "REFUSE" and not res.commands
    second = LAST + 1 + HELD_SPAN
    assert any(r.startswith(f"HELD {second:,}") and "PART (i)" in r for r in res.reasons)
    assert not any(r.startswith(f"HELD {LAST + 1:,}") for r in res.reasons), (
        "the first window passed; only the second should be named")


def test_failing_the_first_window_refuses(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.20, held_cand=[0.97, 0.80])
    assert _gate(tmp_path, flying, candidate).verdict == "REFUSE"


# -- F1: whole orbits, back to back ----------------------------------------------------

def test_every_window_is_the_warm_up_plus_whole_orbits(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.20, held_cand=[0.80, 0.80])
    n = _gate(tmp_path, flying, candidate).numbers
    lo, hi = n["need"]["window"]
    assert (lo, hi) == (LAST + 1 - NEED_SPAN, LAST + 1)
    assert (hi - lo - gate_mod.WARM) % ORBIT == 0
    (a0, a1), (b0, b1) = (w["held"] for w in n["replication"])
    assert (a0, a1 - a0, b0, b1 - b0) == (LAST + 1, HELD_SPAN, a1, HELD_SPAN)
    assert (HELD_SPAN - gate_mod.WARM) % ORBIT == 0


# -- the shadow ------------------------------------------------------------------------

def test_the_shadow_judges_but_never_certifies(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.05, held_cand=[0.80, 0.80])
    res = _gate(tmp_path, flying, candidate, shadow=True)
    assert res.verdict == "NOT NEEDED" and not res.commands
    assert res.numbers["shadow_verdict"] == "WOULD CERTIFY"
    _stub(monkeypatch, need=1.05, held_cand=[0.80, 0.97])
    res = _gate(tmp_path, flying, candidate, shadow=True)
    assert res.numbers["shadow_verdict"] == "WOULD REFUSE"


# -- the baseline is bound to its model ------------------------------------------------

def test_a_baseline_for_another_model_is_refused(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.20, held_cand=[0.80, 0.80])
    with pytest.raises(ToolkitError, match="another flying model"):
        _gate(tmp_path, flying, candidate, baseline=_baseline(flying, sha="0" * 64))
    with pytest.raises(ToolkitError, match="another orbit"):
        _gate(tmp_path, flying, candidate, baseline=_baseline(flying, orbit=ORBIT + 1))


def test_a_need_window_inside_the_baseline_is_refused(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.20, held_cand=[0.80, 0.80])
    with pytest.raises(ToolkitError, match="overlaps the baseline"):
        _gate(tmp_path, flying, candidate, last=NEED_SPAN + 10)


def test_d85b_without_a_baseline_is_refused(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    _stub(monkeypatch, need=1.20, held_cand=[0.80, 0.80])
    with pytest.raises(ToolkitError, match="baseline and the need threshold"):
        gate_mod.gate(
            flying=flying, candidate=candidate, telemetry=np.zeros((60_000, 2), np.float32),
            names=["a", "b"], first_data=24_000, last_data=LAST, windows=6550, budget=1,
            tools=Path("t"), yellow_low=np.array([-10.0, -10.0]),
            yellow_high=np.array([10.0, 10.0]), block=5400, blocks=4, horizon=100_000,
            workdir=tmp_path / "w", orbit=ORBIT)


def test_the_baseline_is_measured_over_whole_orbits_and_bound(tmp_path, monkeypatch) -> None:
    flying, _ = _setup(tmp_path)
    seen = []
    monkeypatch.setattr(gate_mod, "residual",
                        lambda tools, model, rows_file, n, lo, hi, warm=gate_mod.WARM:
                        seen.append((lo, hi)) or 2.5)
    rec = gate_mod.baseline(flying=flying, telemetry=np.zeros((10_000, 2), np.float32),
                            start=100, orbit=ORBIT, tools=Path("t"), workdir=tmp_path / "b")
    assert seen == [(100, 100 + NEED_SPAN)] and rec["residual"] == 2.5
    assert rec["flying_sha256"] == hashlib.sha256(flying.read_bytes()).hexdigest()
    render = gate_mod.render(gate_mod.GateResult(verdict="NOT NEEDED", reasons=["x"]))
    assert "NOT JUDGED BECAUSE" in render
