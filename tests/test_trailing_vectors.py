"""The committed `.tvec` tiers regenerate, and they exercise what they claim to.

Same discipline `tests/test_golden_vectors.py` applies to the `.vec` tiers: a
committed expected value nobody can regenerate is not evidence, it is a number
somebody once wrote down. These are the reference half of `docs/MODELS.md` 39's
N1, so they are held to it on the Python side too.
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

VECTORS = ROOT / "flight" / "test" / "vectors"
TIERS = ("t1", "t2", "t3")
SPAN = 2100


def header(blob: bytes) -> dict:
    assert blob[:4] == b"SNTT", "not a SNTT vector"
    version, channels, steps, span = struct.unpack_from("<HHII", blob, 4)
    return {"version": version, "channels": channels, "steps": steps, "span": span,
            "body": 16 + channels * 4}


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tvec")
    proc = subprocess.run(
        [sys.executable, "scripts/make_trailing_vectors.py", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_the_committed_vector_regenerates_byte_identically(tier, regenerated) -> None:
    assert (VECTORS / f"{tier}.tvec").read_bytes() == (regenerated / f"{tier}.tvec").read_bytes()


@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_runs_past_the_window_so_the_ring_wraps(tier) -> None:
    """A vector shorter than the window never tests the subtraction at all."""
    h = header((VECTORS / f"{tier}.tvec").read_bytes())
    assert h["span"] == SPAN
    assert h["steps"] > SPAN, f"{tier} never wraps: {h['steps']} steps of a {SPAN} window"


def test_the_reference_agrees_with_numpy_on_the_regime_that_broke_before() -> None:
    """T3 is N(1000, 3): D37's regime, where float32 lost 7.66 on a sigma of 3.

    Checked against `numpy.nanstd` directly rather than against the generator, so
    this does not pass by agreeing with the thing it is testing.
    """
    from decision_layer_arms import trailing_stats

    blob = (VECTORS / "t3.tvec").read_bytes()
    h = header(blob)
    record = h["channels"] * 20
    values = np.array([
        struct.unpack_from(f"<{h['channels']}f", blob, h["body"] + t * record)
        for t in range(h["steps"])], dtype=np.float32)

    channel = values[:, 0].astype(np.float64)
    _, sd = trailing_stats(channel, SPAN)
    tail = range(SPAN, h["steps"])
    worst = max(abs(sd[t] - np.nanstd(channel[t + 1 - SPAN:t + 1])) for t in tail)
    assert worst < 1e-9, f"the reference disagrees with numpy.nanstd by {worst:.3e}"


def test_the_tiers_cover_three_widths_and_the_dangerous_regime() -> None:
    widths = {t: header((VECTORS / f"{t}.tvec").read_bytes())["channels"] for t in TIERS}
    assert len(set(widths.values())) == 3, f"tiers duplicate a width: {widths}"
    assert widths["t3"] == 12, "T3 must be the flown width"
