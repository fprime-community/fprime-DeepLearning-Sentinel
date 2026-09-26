"""D85 / `docs/MODELS.md` 78: the ground gate, the segment sidecar and the approval --
their decision logic, each in both directions, with no toolchain.

The gate's arithmetic runs `ground_tools` (the flown cycle and the flight core), which
needs the OxCaml switch; that path is exercised by the arms. What is checked here is
everything the gate DECIDES from those numbers, with the two calls replaced by stubs
that return chosen residuals:

* the sidecar binds a model to its training segment, and refuses any other model;
* integrity: a candidate that is not "the flying file with new weights" is refused;
* part (i) certifies only at or above m = K * F, with K imported;
* a floor wider than 69.7's cold seed spread refuses;
* part (ii) refuses a candidate whose generalisation gap exceeds the flying model's;
* the trend-to-limit table carries NUMBERS for every channel, and a channel heading
  for its limit inside the horizon refuses (the owner's ruling of 2026-09-25);
* `approve` refuses a REFUSE report and a candidate that is not the one certified.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sentinel_toolkit import approve as approve_mod
from sentinel_toolkit import gate as gate_mod
from sentinel_toolkit import segment as segment_mod
from sentinel_toolkit.errors import ToolkitError


def _model(n: int = 400, crc: int = 7) -> bytes:
    """A byte string shaped like a model file's header: cb at 32, wb at 36, crc at 44."""
    b = bytearray(n)
    b[32:36] = (40).to_bytes(4, "little")          # channel records
    b[36:40] = (200).to_bytes(4, "little")         # weights
    b[44:48] = crc.to_bytes(4, "little")
    return bytes(b)


def _setup(tmp_path: Path, rows: int = 7000):
    flying = tmp_path / "flying.bin"
    fb = _model()
    flying.write_bytes(fb)
    train = np.random.default_rng(0).normal(size=(rows, 2)).astype(np.float32)
    segment_mod.write(flying, fb, train, {"window": 250, "n_predictions": 10})
    cand = bytearray(fb)
    cand[64 + 40 + 5] ^= 0xFF                      # inside the weights
    cand[44] ^= 0x01                               # its static crc
    candidate = tmp_path / "cand.bin"
    candidate.write_bytes(bytes(cand))
    return flying, candidate


def _stub(monkeypatch, residuals: dict):
    """Replace the two toolchain calls. `residuals` maps a model name to res values."""
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    def fake_residual(tools, model, rows_file, rows, lo, hi, warm=gate_mod.WARM):
        name = Path(model).name
        if name.startswith("control_"):
            offset = int(name.split("_o")[1].split(".")[0])
            return residuals["controls"][residuals["offsets"].index(offset)]
        if rows_file.name == "segment.f32":
            return residuals["flying_fit"]
        if name == "cand.bin":
            return residuals["cand_held"] if lo > residuals["split"] else residuals["cand_fit"]
        return residuals["flying_held"]
    monkeypatch.setattr(gate_mod, "residual", fake_residual)


class _FakePopen:
    def __init__(self, args, **kw):
        Path(args[-1]).write_bytes(b"control")
        self.returncode = 0

    def communicate(self):
        return ("", "")


def _run(tmp_path, monkeypatch, *, controls, cand_held, cand_fit=None, trend_values=None,
         horizon=100_000):
    flying, candidate = _setup(tmp_path)
    rows = 7000
    room = rows - (6550 + 260 - 1)
    offsets = [k * room // 5 for k in range(6)]
    _stub(monkeypatch, {"controls": controls, "offsets": offsets, "cand_held": cand_held,
                        "cand_fit": cand_held if cand_fit is None else cand_fit,
                        "flying_fit": 1.0, "flying_held": 1.0, "split": 30_000})
    tele = trend_values if trend_values is not None else np.zeros((50_000, 2), dtype=np.float32)
    return gate_mod.gate(
        flying=flying, candidate=candidate, telemetry=tele, names=["a", "b"],
        first_data=24_000, last_data=30_000, windows=6550, budget=1, tools=Path("ground_tools"),
        yellow_low=np.array([-10.0, -10.0]), yellow_high=np.array([10.0, 10.0]),
        block=5400, blocks=4, horizon=horizon, workdir=tmp_path / "w", cache=tmp_path / "c")


# -- the sidecar ------------------------------------------------------------------------

def test_the_sidecar_binds_a_model_to_its_segment(tmp_path: Path) -> None:
    flying, _ = _setup(tmp_path)
    seg = segment_mod.read(flying, flying.read_bytes())
    assert seg.train.shape == (7000, 2)


def test_a_sidecar_for_another_model_is_refused(tmp_path: Path) -> None:
    flying, _ = _setup(tmp_path)
    with pytest.raises(ToolkitError, match="different model"):
        segment_mod.read(flying, _model(crc=8))


def test_a_model_without_a_sidecar_cannot_be_gated(tmp_path: Path) -> None:
    lone = tmp_path / "lone.bin"
    lone.write_bytes(_model())
    with pytest.raises(ToolkitError, match="no training segment"):
        segment_mod.read(lone, lone.read_bytes())


# -- the decision ----------------------------------------------------------------------

GOOD_CONTROLS = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]      # F = 4.0%, m = 8.13%


def test_a_clear_improvement_certifies(tmp_path, monkeypatch) -> None:
    res = _run(tmp_path, monkeypatch, controls=GOOD_CONTROLS, cand_held=0.80)
    assert res.verdict == "CERTIFY", res.reasons
    assert abs(res.numbers["margin_m"] - gate_mod.K * res.numbers["floor_F"]) < 1e-12
    assert len(res.commands) == 2 and "RELOAD_MODEL" in res.commands[1]


def test_an_improvement_below_the_margin_refuses(tmp_path, monkeypatch) -> None:
    res = _run(tmp_path, monkeypatch, controls=GOOD_CONTROLS, cand_held=0.95)
    assert res.verdict == "REFUSE" and any("PART (i)" in r for r in res.reasons)
    assert not res.commands


def test_a_floor_wider_than_the_seed_spread_refuses(tmp_path, monkeypatch) -> None:
    res = _run(tmp_path, monkeypatch, controls=[1.0, 1.4, 0.8, 1.0, 1.0, 1.0], cand_held=0.10)
    assert res.verdict == "REFUSE" and any("FLOOR" in r for r in res.reasons)


def test_a_worse_generalisation_gap_refuses(tmp_path, monkeypatch) -> None:
    """Part (ii): held 0.80 against fit 1.00 is a 20% gap; the flying model's is 0%."""
    res = _run(tmp_path, monkeypatch, controls=GOOD_CONTROLS, cand_held=0.80, cand_fit=1.00)
    assert res.numbers["gap_candidate"] > res.numbers["gap_flying"]
    assert res.verdict == "REFUSE" and any("PART (ii)" in r for r in res.reasons)


def test_integrity_refuses_a_candidate_that_changed_more_than_its_weights(tmp_path, monkeypatch) -> None:
    flying, candidate = _setup(tmp_path)
    bad = bytearray(candidate.read_bytes())
    bad[10] ^= 0xFF                                  # a header byte
    candidate.write_bytes(bytes(bad))
    res = gate_mod.gate(
        flying=flying, candidate=candidate, telemetry=np.zeros((50_000, 2), np.float32),
        names=["a", "b"], first_data=24_000, last_data=30_000, windows=6550, budget=1,
        tools=Path("ground_tools"), yellow_low=np.array([-10.0, -10.0]),
        yellow_high=np.array([10.0, 10.0]), block=5400, blocks=4, horizon=100_000,
        workdir=tmp_path / "w")
    assert res.verdict == "REFUSE" and "INTEGRITY" in res.reasons[0]


# -- the trend, and the owner's ruling that it shows the numbers ------------------------

def _ramp(slope_per_tick: float) -> np.ndarray:
    t = np.arange(50_000, dtype=np.float64)
    wave = np.sin(2 * np.pi * t / 5400.0)
    return np.stack([wave + slope_per_tick * t, wave], axis=1).astype(np.float32)


def test_a_channel_heading_for_its_limit_inside_the_horizon_refuses(tmp_path, monkeypatch) -> None:
    # 1e-4 per tick from about 3 toward 10: about 70,000 ticks, inside 100,000
    res = _run(tmp_path, monkeypatch, controls=GOOD_CONTROLS, cand_held=0.80,
               trend_values=_ramp(1.0e-4))
    row = next(t for t in res.trend if t["channel"] == "a")
    assert row["inside_horizon"] and row["ticks_to_yellow"] < 100_000
    assert any("TREND TO LIMIT" in r for r in res.reasons)


def test_a_flat_channel_is_not_refused_and_every_channel_carries_numbers(tmp_path, monkeypatch) -> None:
    res = _run(tmp_path, monkeypatch, controls=GOOD_CONTROLS, cand_held=0.80,
               trend_values=_ramp(0.0))
    assert not any("TREND" in r for r in res.reasons)
    assert len(res.trend) == 2
    for t in res.trend:
        for key in ("slope_of_maxima_per_tick", "slope_of_minima_per_tick", "ticks_to_yellow",
                    "horizon", "margin_ticks", "last_block_max", "last_block_min"):
            assert key in t
    text = gate_mod.render(res)
    assert "TREND TO LIMIT, EVERY CHANNEL" in text and "ticks to yellow" in text


# -- approve ----------------------------------------------------------------------------

def test_approve_refuses_a_refuse_report(tmp_path: Path) -> None:
    cand = tmp_path / "c.bin"
    cand.write_bytes(b"x")
    with pytest.raises(ToolkitError, match="only a CERTIFY report"):
        approve_mod.check({"verdict": "REFUSE", "reasons": ["PART (i)"]}, cand)


def test_approve_refuses_a_candidate_that_was_not_the_one_certified(tmp_path: Path) -> None:
    cand = tmp_path / "c.bin"
    cand.write_bytes(b"x")
    report = {"verdict": "CERTIFY", "numbers": {"candidate_sha256": "0" * 64},
              "commands": ["a", "b"]}
    with pytest.raises(ToolkitError, match="not the candidate the gate certified"):
        approve_mod.check(report, cand)


def test_approve_dry_run_names_the_commands_and_sends_nothing(tmp_path: Path, capsys) -> None:
    import hashlib
    cand = tmp_path / "c.bin"
    cand.write_bytes(b"candidate")
    report = tmp_path / "r.json"
    report.write_text(json.dumps({
        "verdict": "CERTIFY",
        "numbers": {"candidate_sha256": hashlib.sha256(b"candidate").hexdigest(),
                    "uplink_destination": "RetrainApproved.bin",
                    "part_i_improvement": 0.2, "margin_m": 0.1},
        "commands": ["fprime-cli file-uplink c.bin RetrainApproved.bin",
                     "fprime-cli command-send X.RELOAD_MODEL --arguments RetrainApproved.bin"]}))
    assert approve_mod.approve(report, cand, None, dry_run=True, log=print) == 0
    out = capsys.readouterr().out
    assert "would run" in out and "RELOAD_MODEL" in out
