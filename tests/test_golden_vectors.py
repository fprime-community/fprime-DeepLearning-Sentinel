"""The golden vectors: they cannot drift, and they are what `reference.py` says.

Two obligations, and they are different. The first is that a committed vector is
exactly what the generator produces today, so nobody can quietly edit an expected
number to make a failing C++ build pass. The second is that the generator's output
is genuinely `reference.forward` plus the frozen decision layer (D25), so the
vectors are worth pinning in the first place.

The C++ half -- that the flight core reproduces these files -- lives in
`flight/test/GoldenVectors.cpp` and is run by `make -C flight test`.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_golden_vectors as gen                       # noqa: E402
from sentinel_export import Status, read_model          # noqa: E402
from sentinel_models import reference, telemanom        # noqa: E402
from sentinel_models.windows import aggregate_predictions  # noqa: E402

VECTORS = ROOT / "flight" / "test" / "vectors"

#: The tolerance that holds the NumPy reference to torch
#: (`tests/test_reference_equivalence.py:36`), and the tolerance the C++ is held
#: to against the reference (`docs/MODELS.md` 19, prediction F5).
TOLERANCE = 1e-5

COMMITTED = ("g1", "g2", "g3")


def parse(path: Path) -> dict:
    """Read a `.vec` file back into arrays. Mirrors `flight/test/GoldenVectors.cpp`."""
    raw = path.read_bytes()
    head = gen.VEC_HEADER.unpack(raw[:gen.VEC_HEADER.size])
    (magic, version, n_channels, n_layers, h0, h1, h2, h3, n_predictions,
     n_steps, phase1_end, phase2_end, ewma_span, warmup, threshold) = head
    assert magic == gen.VEC_MAGIC and version == gen.VEC_VERSION

    hidden = [h for h in (h0, h1, h2, h3)][:n_layers]
    cursor = gen.VEC_HEADER_BYTES

    def take(rows, cols):
        nonlocal cursor
        count = rows * cols
        out = np.frombuffer(raw, dtype="<f4", count=count, offset=cursor)
        cursor += 4 * count
        return np.array(out, dtype=np.float32).reshape(rows, cols)

    out = {
        "n_channels": n_channels, "hidden": hidden, "n_predictions": n_predictions,
        "n_steps": n_steps, "phase1_end": phase1_end, "phase2_end": phase2_end,
        "ewma_span": ewma_span, "warmup": warmup, "threshold": threshold,
        "input": take(n_steps, n_channels),
        "hidden_trace": take(n_steps, sum(hidden)),
        "head": take(n_steps, n_predictions * n_channels),
        "forecast": take(n_steps, n_channels),
        "residual": take(n_steps, n_channels),
        "ewma": take(n_steps, n_channels),
    }
    out["score"] = take(n_steps, 1).reshape(n_steps)
    out["crossing"] = np.frombuffer(raw, dtype=np.uint8, count=n_steps,
                                    offset=cursor).astype(bool)
    cursor += n_steps
    out["emitted"] = np.frombuffer(raw, dtype=np.uint8, count=n_steps,
                                   offset=cursor).astype(bool)
    cursor += n_steps
    assert cursor == len(raw), f"{path.name}: {len(raw) - cursor} trailing bytes"
    return out


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """Regenerate every committed tier into a temp directory. Never the worktree."""
    out = tmp_path_factory.mktemp("vectors")
    gen.set_output(out)
    try:
        for tier, builder in (
                ("g1", lambda: (gen.seeded_weights(3, [4, 4], 2, seed=1),
                                gen.seeded_input(gen.P3, 3, seed=101),
                                "seeded G1, rng(1), U(-k,k) k=1/sqrt(H)")),
                ("g2", lambda: (gen.seeded_weights(7, [24, 24], 5, seed=2),
                                gen.fixture_input(gen.P3),
                                "seeded G2 on synthetic.build(seed=0, n=8000)")),
                ("g3", lambda: (gen.seeded_weights(12, [80, 80], 10, seed=3),
                                gen.seeded_input(gen.P3, 12, seed=103),
                                "seeded G3 at the flown shape, rng(3)"))):
            weights, x, provenance = builder()
            gen.emit(tier, weights, x, provenance)
    finally:
        gen.set_output(VECTORS)
    return out


# -- they cannot drift ------------------------------------------------------

@pytest.mark.parametrize("tier", COMMITTED)
def test_the_committed_vector_is_byte_identical_to_a_fresh_generation(tier, regenerated):
    """A committed expected value nobody can regenerate is not evidence."""
    committed = (VECTORS / f"{tier}.vec").read_bytes()
    fresh = (regenerated / f"{tier}.vec").read_bytes()
    assert len(committed) == len(fresh), f"{tier}.vec changed size"
    assert committed == fresh, f"{tier}.vec differs from a fresh generation"


@pytest.mark.parametrize("tier", ("g1", "g2"))
def test_the_committed_model_file_is_byte_identical_to_a_fresh_generation(tier, regenerated):
    """G3's 285 KB weight file regenerates from its seed and is not committed."""
    assert (VECTORS / f"{tier}.bin").read_bytes() == (regenerated / f"{tier}.bin").read_bytes()


def test_g3_weights_are_not_committed_and_regenerate_from_the_seed(regenerated):
    fresh = (regenerated / "g3.bin")
    assert fresh.stat().st_size == 285_136, "the flown shape is 285,136 bytes"
    status, _ = read_model(fresh.read_bytes())
    assert status is Status.OK


def test_the_manifest_records_only_the_tiers_a_fresh_clone_can_rebuild():
    """G4 is built from cached weights under `runs/`, which is gitignored, so its
    rows must not reach the committed manifest -- a fresh clone would find a
    manifest naming files it cannot produce."""
    rows = json.loads((VECTORS / "manifest.json").read_text())
    assert [row["tier"] for row in rows] == list(COMMITTED)
    for row in rows:
        assert (VECTORS / f"{row['tier']}.vec").stat().st_size == row["vector_bytes"]


# -- they are what the reference says ---------------------------------------

@pytest.mark.parametrize("tier, n_channels, hidden, l_p, seed", [
    ("g1", 3, [4, 4], 2, 1),
    ("g2", 7, [24, 24], 5, 2),
    ("g3", 12, [80, 80], 10, 3),
])
def test_the_vector_is_reference_forward_and_the_frozen_decision_layer(
        tier, n_channels, hidden, l_p, seed):
    """Re-derive every stage independently of the generator's own bookkeeping."""
    vector = parse(VECTORS / f"{tier}.vec")
    weights = gen.seeded_weights(n_channels, hidden, l_p, seed=seed)
    x = vector["input"]
    p1, p2, p3 = vector["phase1_end"], vector["phase2_end"], vector["n_steps"]

    # Phases A and B are one continuous stream; C starts over, as reset() does.
    stream_ab, _ = reference.forward(weights, x[:p2][None, ...])
    stream_c, _ = reference.forward(weights, x[p2:p3][None, ...])
    predictions = np.concatenate([stream_ab[0], stream_c[0]], axis=0)
    assert np.allclose(vector["head"],
                       predictions.reshape(p3, l_p * n_channels), atol=TOLERANCE)

    forecast, residual, smoothed, score = [], [], [], []
    for chunk_x, chunk_p in ((x[:p2], stream_ab[0]), (x[p2:p3], stream_c[0])):
        f = aggregate_predictions(chunk_p)
        r = np.abs(chunk_x - f).astype(np.float32)
        s = telemanom.ewma(r, 105, telemanom.EwmaState())
        forecast.append(f)
        residual.append(r)
        smoothed.append(s)
        score.append(s.max(axis=1).astype(np.float32))

    assert np.allclose(vector["forecast"], np.concatenate(forecast), atol=TOLERANCE)
    assert np.allclose(vector["residual"], np.concatenate(residual), atol=TOLERANCE)
    assert np.allclose(vector["ewma"], np.concatenate(smoothed), atol=TOLERANCE)
    assert np.allclose(vector["score"], np.concatenate(score), atol=TOLERANCE)


@pytest.mark.parametrize("tier", COMMITTED)
def test_the_crossing_flag_is_the_score_against_the_threshold_in_float64(tier):
    """`harness.py:169-172`: `>=`, not `>`, and compared in float64."""
    vector = parse(VECTORS / f"{tier}.vec")
    expected = vector["score"].astype(np.float64) >= vector["threshold"]
    assert np.array_equal(vector["crossing"], expected)


@pytest.mark.parametrize("tier", COMMITTED)
def test_the_flag_actually_flips_so_the_assertion_is_not_vacuous(tier):
    """A vector whose crossing flag never changes tests nothing about it."""
    crossing = parse(VECTORS / f"{tier}.vec")["crossing"]
    assert 0 < crossing.sum() < len(crossing), f"{tier}: flag is constant"


@pytest.mark.parametrize("tier", COMMITTED)
def test_the_warm_up_gate_is_silent_and_reopens_after_a_reset(tier):
    """Objective.md 11 rule 2: no output until enough history backs a warning."""
    vector = parse(VECTORS / f"{tier}.vec")
    warmup, p2 = vector["warmup"], vector["phase2_end"]
    assert not vector["emitted"][:warmup].any(), "emitted during the opening warm-up"
    assert not vector["emitted"][p2:p2 + warmup].any(), (
        "a reset must restart the warm-up, not inherit the old one")
    outside = np.ones(vector["n_steps"], dtype=bool)
    outside[:warmup] = False
    outside[p2:p2 + warmup] = False
    assert np.array_equal(vector["emitted"][outside], vector["crossing"][outside])


@pytest.mark.parametrize("tier", COMMITTED)
def test_the_state_really_is_carried_across_the_chunk_boundary(tier):
    """Phase B continues phase A rather than restarting.

    Two-sided, because a one-sided check passes on a vector built either way: the
    recorded phase-B hidden state must match the carried continuation, and must
    **differ** from the same steps run from a zero state. If it did not differ,
    the test would be measuring nothing and the chunk boundary would be untested.
    """
    vector = parse(VECTORS / f"{tier}.vec")
    p1, p2 = vector["phase1_end"], vector["phase2_end"]
    weights = _weights_for(tier)
    x = vector["input"]
    width = weights.layers[0].hidden

    _, carried = reference.gru_layer(x[:p1][None, ...], weights.layers[0],
                                     reference.zero_state(weights, 1)[0])
    continued, _ = reference.gru_layer(x[p1:p2][None, ...], weights.layers[0], carried)
    restarted, _ = reference.gru_layer(x[p1:p2][None, ...], weights.layers[0],
                                       reference.zero_state(weights, 1)[0])

    recorded = vector["hidden_trace"][p1:p2, :width]
    assert np.allclose(recorded, continued[0], atol=TOLERANCE), (
        "phase B is not the carried continuation of phase A")
    assert np.abs(continued[0] - restarted[0]).max() > TOLERANCE, (
        "a zero-state restart is indistinguishable here, so this proves nothing")
    assert not np.allclose(recorded, restarted[0], atol=TOLERANCE), (
        "phase B looks like a restart -- the state was dropped")


@pytest.mark.parametrize("tier", COMMITTED)
def test_the_reset_really_does_start_from_a_zero_state(tier):
    """The first step after the reset must be what a cold start gives, and must
    differ from what continuing phase B would have given."""
    vector = parse(VECTORS / f"{tier}.vec")
    p1, p2 = vector["phase1_end"], vector["phase2_end"]
    weights = _weights_for(tier)
    x = vector["input"]
    width = weights.layers[0].hidden

    cold, _ = reference.gru_layer(x[p2:p2 + 1][None, ...], weights.layers[0],
                                  reference.zero_state(weights, 1)[0])
    _, warm = reference.gru_layer(x[:p2][None, ...], weights.layers[0],
                                  reference.zero_state(weights, 1)[0])
    uncut, _ = reference.gru_layer(x[p2:p2 + 1][None, ...], weights.layers[0], warm)

    recorded = vector["hidden_trace"][p2, :width]
    assert np.allclose(recorded, cold[0, 0], atol=TOLERANCE), "the reset did not zero the state"
    assert np.abs(cold[0, 0] - uncut[0, 0]).max() > TOLERANCE, (
        "a reset is indistinguishable from continuing here, so this proves nothing")


def _weights_for(tier: str):
    shape = {"g1": (3, [4, 4], 2, 1), "g2": (7, [24, 24], 5, 2),
             "g3": (12, [80, 80], 10, 3)}[tier]
    return gen.seeded_weights(shape[0], shape[1], shape[2], seed=shape[3])


# -- the model files the C++ loads -----------------------------------------

@pytest.mark.parametrize("tier", ("g1", "g2"))
def test_the_committed_model_file_loads_and_matches_its_vector(tier):
    vector = parse(VECTORS / f"{tier}.vec")
    status, spec = read_model((VECTORS / f"{tier}.bin").read_bytes())
    assert status is Status.OK
    assert spec["n_channels"] == vector["n_channels"]
    assert spec["hidden"] == vector["hidden"]
    assert spec["n_predictions"] == vector["n_predictions"]
    assert spec["params"]["threshold"] == vector["threshold"]
    assert spec["params"]["ewma_span"] == vector["ewma_span"]
    assert spec["params"]["warmup_steps"] == vector["warmup"]
