"""The streamed rule is a restatement of `telemanom`, not a second opinion.

`src/sentinel_models/flight_reference.py` states the dynamic threshold as the
flight component computes it: one tick at a time, deciding at the end of each
segment with nothing after it, and dilating forward only. Two of those three are
departures (`docs/MODELS.md` 39.5), and a departure is only honest if the thing it
departs from is reproduced exactly when the departure is switched off.

**That is what the first test does.** With backward dilation restored, the
streamed form must produce byte-identical emission arrays to
`telemanom.channel_ratios`, which is the published rule this project has measured
everything on.
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sentinel_models.flight_reference import dilate, emissions, solve_window
from sentinel_models.telemanom import Config, _buffered, channel_ratios

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "flight" / "test" / "vectors"
TIERS = ("d1", "d2")


def series(seed: int, steps: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = np.abs(rng.normal(0.0, 1.0, steps)).astype(np.float32)
    e[steps // 3:steps // 3 + 30] *= 15.0
    e[2 * steps // 3:2 * steps // 3 + 8] *= 6.0
    return e


@pytest.mark.parametrize("steps", [70 * 40, 70 * 45])
@pytest.mark.parametrize("seed", [5, 17])
def test_the_streamed_form_reproduces_channel_ratios_exactly(seed, steps) -> None:
    """The faithfulness claim, and the whole licence for the departures."""
    e_s = series(seed, steps)
    reference = np.zeros(steps, dtype=bool)
    channel_ratios(e_s, Config(), emission=reference)
    streamed, _ = emissions(e_s, Config(), forward_only=False)
    assert np.array_equal(reference, streamed), (
        f"streamed emission differs at "
        f"{np.flatnonzero(reference != streamed)[:8].tolist()}")


def test_forward_only_dilation_is_buffered_without_the_backward_pad() -> None:
    mask = np.zeros(500, dtype=bool)
    mask[200:205] = True
    both = _buffered(mask, Config().error_buffer)
    forward = dilate(mask, Config().error_buffer, forward_only=True)
    assert len(both) == len(forward) == 1
    assert both[0][1] == forward[0][1], "the forward edge is unchanged"
    assert forward[0][0] == 200, "the run starts where the crossing did"
    assert both[0][0] == 101, "the reference reaches 99 steps back"


def test_dropping_the_backward_pad_never_widens_a_sequence() -> None:
    """It can only ever remove alarm extent, never add it."""
    rng = np.random.default_rng(2)
    mask = rng.random(3000) > 0.98
    both = _buffered(mask, Config().error_buffer)
    forward = dilate(mask, Config().error_buffer, forward_only=True)
    covered_both = sum(hi - lo for lo, hi in both)
    covered_forward = sum(hi - lo for lo, hi in forward)
    assert covered_forward <= covered_both


def test_a_flat_channel_stays_silent_rather_than_guessing() -> None:
    flat = np.full(70 * 10, 0.25, dtype=np.float32)
    emitted, _ = emissions(flat, Config())
    assert not emitted.any(), "zero variance must produce silence, not a guess"


def test_the_held_threshold_lags_by_exactly_one_segment() -> None:
    e_s = series(9, 70 * 12)
    _, held = emissions(e_s, Config())
    stride = Config().stride
    assert np.all(held[:stride] == 0.0), "nothing is in force before the first solve"
    for seg in range(1, 12):
        block = held[seg * stride:(seg + 1) * stride]
        assert len(set(block.tolist())) == 1, "the threshold is constant within a segment"


def test_solve_window_delegates_to_the_published_sweep_when_it_can() -> None:
    """With backward dilation on, the sweep must be telemanom's own, not a copy."""
    from sentinel_models.telemanom import dynamic_threshold
    window = series(4, 2170)
    mine = solve_window(window, Config(), forward_only=False)
    eps, sequences = dynamic_threshold(window, Config())
    assert mine.eps == float(eps)
    assert mine.sequences == sequences


# -- the committed vectors ----------------------------------------------------

@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dvec")
    proc = subprocess.run(
        [sys.executable, "scripts/make_threshold_vectors.py", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_the_committed_vector_regenerates_byte_identically(tier, regenerated) -> None:
    assert (VECTORS / f"{tier}.dvec").read_bytes() == (regenerated / f"{tier}.dvec").read_bytes()


@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_runs_past_the_solve_window(tier) -> None:
    blob = (VECTORS / f"{tier}.dvec").read_bytes()
    assert blob[:4] == b"SNTD"
    _, _, steps, span, stride, solve = struct.unpack_from("<HHIIII", blob, 4)
    assert solve == span + stride, "the solve window is history plus the segment"
    assert steps > solve, f"{tier} never fills its solve window"


@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_actually_emits(tier) -> None:
    """A vector on which the rule never fires proves nothing about the rule."""
    blob = (VECTORS / f"{tier}.dvec").read_bytes()
    channels, steps = struct.unpack_from("<HI", blob, 6)[0], struct.unpack_from("<I", blob, 8)[0]
    record = channels * 12 + 1
    body = 24
    fired = sum(blob[body + t * record + record - 1] for t in range(steps))
    assert fired > 0, f"{tier} never emits"
