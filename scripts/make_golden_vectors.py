"""Generate the golden vectors the C++ core is held to, and the files it loads.

Four tiers (`docs/MODELS.md` 19.6). Every one is a three-phase scenario that
exercises the two things a stateful flight loop can get wrong and a single-shot
test cannot see:

    phase A   steps [0, P1)     from a zero state
    phase B   steps [P1, P2)    continuing with the carried state  -- chunk boundary
    phase C   steps [P2, P3)    after a reset, from a zero state again

Phases A and B are one continuous stream, which is legitimate because
`tests/test_reference_equivalence.py:260` pins that a chunked call reproduces the
whole. Phase C restarts the GRU state, the prediction ring and the EWMA together,
which is what `reset()` means in flight.

This script is the bridge the format writer deliberately does not have. Nothing in
`src/sentinel_export/` imports `sentinel_models`, so the conversion from a
`reference.Weights` to the plain dictionary the writer takes lives here.

Zero bucket operations. Every input is the seeded fixture or a seeded generator;
no golden vector uses real telemetry, because none is on local disk.
"""
from __future__ import annotations

import argparse
import glob
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_eval import bundle as bundle_mod          # noqa: E402
from sentinel_eval import read, splits, synthetic, tasks  # noqa: E402
from sentinel_eval.catalog import Catalog                # noqa: E402
from sentinel_eval.labels import LabelSet                # noqa: E402
from sentinel_export import format as fmt                # noqa: E402
from sentinel_export import write_model                  # noqa: E402
from sentinel_models import reference, telemanom         # noqa: E402
from sentinel_models.windows import aggregate_predictions  # noqa: E402

VECTORS = ROOT / "flight" / "test" / "vectors"


def set_output(path) -> None:
    """Point generation somewhere else. `tests/test_golden_vectors.py` regenerates
    into a temporary directory and compares bytes, so the anti-drift check never
    writes into the working tree."""
    global VECTORS
    VECTORS = Path(path)

#: 24 steps a phase. Long enough that the EWMA and the ten-deep prediction ring are
#: both past their opening and into their steady behaviour, short enough that the
#: committed trace stays small.
PHASE = 24
P1, P2, P3 = PHASE, 2 * PHASE, 3 * PHASE

#: The vector fixtures carry a short warm-up so the silent-until-validated gate
#: actually opens inside a 72-step trace. The flown value is 2,350
#: (`detectors.ForecastDetector.warmup_steps`) and is pinned by
#: `tests/test_model_file.py`, not here.
FIXTURE_WARMUP = 8

VEC_MAGIC = b"SNTV"
VEC_VERSION = 1
VEC_HEADER = struct.Struct("<4s13Id")
#: 4 + 13*4 + 8 = 64 exactly: magic, version, n_channels, n_layers, four hidden
#: slots, n_predictions, n_steps, phase1_end, phase2_end, ewma_span,
#: warmup_steps, then the threshold as F64.
VEC_HEADER_BYTES = 64


# -- weights ----------------------------------------------------------------

def seeded_weights(n_channels, hidden, n_predictions, n_exogenous=0, seed=0):
    """PyTorch's own GRU initialiser, by the recipe in `docs/MODEL_FILE.md` 10.

    Arrays are drawn in `reference.Weights.arrays()` order from one generator, so
    the sequence is fixed by the seed and the shape list alone and a committed
    vector regenerates on a fresh clone.
    """
    rng = np.random.default_rng(seed)
    n_inputs = n_channels + n_exogenous
    owner = {}
    for i, h in enumerate(hidden):
        for suffix in fmt.ARRAY_SUFFIXES:
            owner[f"l{i}_{suffix}"] = h
    owner["head_w"] = owner["head_b"] = hidden[-1]

    drawn = {}
    for name, shape in fmt.array_shapes(n_inputs, hidden, n_predictions, n_channels):
        k = 1.0 / np.sqrt(owner[name])
        drawn[name] = rng.uniform(-k, k, shape).astype(np.float32)

    layers = tuple(
        reference.LayerWeights(drawn[f"l{i}_w_ih"], drawn[f"l{i}_w_hh"],
                               drawn[f"l{i}_b_ih"], drawn[f"l{i}_b_hh"])
        for i in range(len(hidden)))
    return reference.Weights(layers=layers, head_w=drawn["head_w"],
                             head_b=drawn["head_b"], n_channels=n_channels,
                             window=250, n_predictions=n_predictions,
                             n_exogenous=n_exogenous)


def cached_production_weights():
    """Every cached GRU fit at the flown shape, newest filename order.

    The weight-cache key is a digest that includes a strided sample of the
    telemetry (`detectors._digest`), which is not on local disk, so a cached file
    cannot be mapped back to a named fold. All of them are used and each is named
    by its store filename (`docs/MODELS.md` 19.6).
    """
    found = []
    for path in sorted(glob.glob(str(ROOT / "runs" / "_weights" / "*.npz"))):
        try:
            blob = np.load(path, allow_pickle=False)
        except Exception:
            continue
        if "l0_w_hh" not in blob.files:
            continue
        hidden_size = int(blob["l0_w_hh"].shape[1])
        gates = blob["l0_w_hh"].shape[0] // max(hidden_size, 1)
        if gates != 3 or int(blob["n_channels"]) != 12 or hidden_size != 80:
            continue
        if int(blob["n_predictions"]) != 10 or int(blob["n_layers"]) != 2:
            continue
        layers = tuple(
            reference.LayerWeights(blob[f"l{i}_w_ih"], blob[f"l{i}_w_hh"],
                                   blob[f"l{i}_b_ih"], blob[f"l{i}_b_hh"])
            for i in range(int(blob["n_layers"])))
        weights = reference.Weights(
            layers=layers, head_w=blob["head_w"], head_b=blob["head_b"],
            n_channels=int(blob["n_channels"]), window=int(blob["window"]),
            n_predictions=int(blob["n_predictions"]),
            n_exogenous=int(blob["n_exogenous"]) if "n_exogenous" in blob.files else 0)
        found.append((Path(path).name, weights))
    return found


# -- inputs -----------------------------------------------------------------

def seeded_input(n_steps, n_channels, seed):
    """A bounded, smoothly varying signal. Not telemetry, and not pretending to be.

    ESA-ADB arrives min-max scaled within each channel group (D2), so a fixture
    input in [0, 1] with a slow component and a fast one is the right order of
    magnitude for the arithmetic under test.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64)[:, None]
    phase = rng.uniform(0.0, 2.0 * np.pi, (1, n_channels))
    slow = 0.5 + 0.25 * np.sin(2.0 * np.pi * t / 37.0 + phase)
    fast = 0.05 * np.sin(2.0 * np.pi * t / 5.0 + 2.0 * phase)
    noise = 0.01 * rng.standard_normal((n_steps, n_channels))
    return np.clip(slow + fast + noise, 0.0, 1.0).astype(np.float32)


def fixture_input(n_steps):
    """The real synthetic fixture, `synthetic.build(seed=0, n=8000)`.

    The same bucket the test suite runs against: real parquet bytes under real
    manifest keys with real SHA-256 digests, so this is the genuine load path and
    not a hand-made array. Seven numeric channels; `channel_8` is categorical and
    the bundle refuses it.
    """
    bucket = synthetic.build(seed=0, n=8_000)
    catalog = Catalog.load(bucket)
    labels = LabelSet.from_table(read.read_annotation(bucket, catalog, "labels"))
    channel_ids = tasks.get("synthetic").selection.resolve(catalog)
    loaded = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                             channel_ids=channel_ids)
    values = loaded.values
    observed = np.isfinite(values).all(axis=1)
    # The first fully observed run long enough for the trace, so no NaN reaches
    # the arithmetic and the vector is about the GRU rather than about gap fill.
    run = 0
    for end in range(len(observed)):
        run = run + 1 if observed[end] else 0
        if run >= n_steps:
            return np.ascontiguousarray(values[end - n_steps + 1:end + 1],
                                        dtype=np.float32)
    raise RuntimeError(f"no fully observed run of {n_steps} steps in the fixture")


# -- the reference pipeline, phase by phase ---------------------------------

def run_stream(weights, x, state=None):
    """One continuous stream: per-step hidden state, head output, predictions.

    `reference.forward` returns only the final state, and a golden vector that
    could not localise a divergence to a layer and a step would not be worth
    committing -- so the layers are driven directly and the result is asserted
    against `forward` before anything is written.
    """
    if state is None:
        state = reference.zero_state(weights, 1)
    activations = x[None, ...].astype(np.float32)
    per_layer, out_state = [], []
    for layer, layer_state in zip(weights.layers, state):
        activations, next_state = reference.gru_layer(activations, layer, layer_state)
        per_layer.append(np.array(activations[0], dtype=np.float32))
        out_state.append(next_state)

    hidden_last = weights.layers[-1].hidden
    flat = (activations.reshape(-1, hidden_last) @ weights.head_w.T + weights.head_b)
    head = np.array(flat, dtype=np.float32)
    predictions = head.reshape(len(x), weights.n_predictions, weights.n_channels)

    expected, _ = reference.forward(weights, x[None, ...], state)
    assert np.array_equal(predictions[None, ...], expected), (
        "the per-layer drive diverged from reference.forward")
    return per_layer, head, predictions, out_state


def decide(x, predictions):
    """The frozen decision layer (D25), on one continuous stream.

    Forecast is the mean of the up-to-ten predictions made at t-1 .. t-10
    (float64 accumulate, float32 out); residual is the absolute per-channel
    difference; EWMA span 105 bias-corrected in float64; max across channels.
    """
    forecast = aggregate_predictions(predictions)
    residual = np.abs(x - forecast).astype(np.float32)
    smoothed = telemanom.ewma(residual, 105, telemanom.EwmaState())
    score = smoothed.max(axis=1).astype(np.float32)
    return forecast, residual, smoothed, score


def build_trace(weights, x):
    """The three-phase scenario. Returns every stage, concatenated over 3 * PHASE."""
    hidden_a, head_a, pred_a, state_a = run_stream(weights, x[:P1])
    hidden_b, head_b, pred_b, _ = run_stream(weights, x[P1:P2], state_a)
    hidden_c, head_c, pred_c, _ = run_stream(weights, x[P2:P3])   # after the reset

    hidden = [np.concatenate([a, b, c], axis=0)
              for a, b, c in zip(hidden_a, hidden_b, hidden_c)]
    head = np.concatenate([head_a, head_b, head_c], axis=0)

    # A and B are one stream; C starts over, exactly as reset() does in flight.
    f1, r1, s1, sc1 = decide(x[:P2], np.concatenate([pred_a, pred_b], axis=0))
    f2, r2, s2, sc2 = decide(x[P2:P3], pred_c)

    return {
        "input": x,
        "hidden": hidden,
        "head": head,
        "forecast": np.concatenate([f1, f2], axis=0),
        "residual": np.concatenate([r1, r2], axis=0),
        "ewma": np.concatenate([s1, s2], axis=0),
        "score": np.concatenate([sc1, sc2], axis=0),
    }


# -- writing ----------------------------------------------------------------

def write_vector(path, weights, trace, threshold, warmup):
    hidden_sizes = [layer.hidden for layer in weights.layers]
    slots = hidden_sizes + [0] * (fmt.HEADER_LAYER_SLOTS - len(hidden_sizes))
    header = VEC_HEADER.pack(
        VEC_MAGIC, VEC_VERSION, weights.n_channels, len(weights.layers),
        *slots, weights.n_predictions, P3, P1, P2, 105, warmup, threshold)
    header = header.ljust(VEC_HEADER_BYTES, b"\x00")

    crossing = (trace["score"].astype(np.float64) >= threshold)
    emitted = crossing.copy()
    emitted[:warmup] = False
    emitted[P2:P2 + warmup] = False          # the reset restarts the warm-up

    body = b"".join([
        trace["input"].astype("<f4").tobytes(),
        np.concatenate(trace["hidden"], axis=1).astype("<f4").tobytes(),
        trace["head"].astype("<f4").tobytes(),
        trace["forecast"].astype("<f4").tobytes(),
        trace["residual"].astype("<f4").tobytes(),
        trace["ewma"].astype("<f4").tobytes(),
        trace["score"].astype("<f4").tobytes(),
        crossing.astype(np.uint8).tobytes(),
        emitted.astype(np.uint8).tobytes(),
    ])
    path.write_bytes(header + body)
    return crossing, emitted


#: (!) MOVED, 2026-09-11. `to_spec` lived here because `sentinel_export` imports
#: nothing from this repository by design, so the bridge from `reference.Weights`
#: to the writer's dict could not live inside it. It could not stay here either:
#: a shipped entry point under `src/` cannot import from `scripts/`. It is now
#: `sentinel_toolkit.spec.to_spec` and this is the same function, not a copy --
#: one definition behind every committed vector (`docs/MODELS.md` 40.7).
#:
#: The one change at the call site is that `param_version` is now REQUIRED. It
#: was omitted here and the writer defaulted it to 1, which is right for these
#: tiers and is now said rather than assumed (40.6).
from sentinel_toolkit.spec import to_spec                             # noqa: E402


def emit(tier, weights, x, provenance, names=None, write_model_file=True):
    trace = build_trace(weights, x)
    # A threshold at the median of the settled score, so the crossing flag flips
    # inside the trace and "exactly on the crossing flag" is a real assertion
    # rather than an all-false one.
    threshold = float(np.median(trace["score"][FIXTURE_WARMUP:].astype(np.float64)))
    VECTORS.mkdir(parents=True, exist_ok=True)

    if write_model_file:
        spec = to_spec(weights, threshold, FIXTURE_WARMUP, provenance, names,
                       param_version=1)
        (VECTORS / f"{tier}.bin").write_bytes(write_model(spec))

    crossing, emitted = write_vector(VECTORS / f"{tier}.vec", weights, trace,
                                     threshold, FIXTURE_WARMUP)
    return {
        "tier": tier,
        "n_channels": weights.n_channels,
        "hidden": [layer.hidden for layer in weights.layers],
        "n_predictions": weights.n_predictions,
        "parameters": weights.n_parameters,
        "steps": P3,
        "threshold": threshold,
        "crossings": int(crossing.sum()),
        "emitted": int(emitted.sum()),
        "vector_bytes": (VECTORS / f"{tier}.vec").stat().st_size,
        "model_bytes": (VECTORS / f"{tier}.bin").stat().st_size if write_model_file else None,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", choices=["g1", "g2", "g3", "g4"],
                        help="default: g1 g2 g3, plus g4 when weights are cached")
    parser.add_argument("--out", default=None,
                        help="write elsewhere than flight/test/vectors")
    args = parser.parse_args()
    if args.out:
        set_output(args.out)
    wanted = set(args.tier or ["g1", "g2", "g3", "g4"])
    report = []

    if "g1" in wanted:
        w = seeded_weights(3, [4, 4], 2, seed=1)
        report.append(emit("g1", w, seeded_input(P3, 3, seed=101),
                           "seeded G1, rng(1), U(-k,k) k=1/sqrt(H)"))

    if "g2" in wanted:
        w = seeded_weights(7, [24, 24], 5, seed=2)
        report.append(emit("g2", w, fixture_input(P3),
                           "seeded G2 on synthetic.build(seed=0, n=8000)"))

    if "g3" in wanted:
        w = seeded_weights(12, [80, 80], 10, seed=3)
        report.append(emit("g3", w, seeded_input(P3, 12, seed=103),
                           "seeded G3 at the flown shape, rng(3)"))

    if "g4" in wanted:
        cached = cached_production_weights()
        if not cached:
            print("g4: no cached production weights under runs/_weights -- skipped")
        for index, (name, w) in enumerate(cached):
            report.append(emit(f"g4_{index}", w, seeded_input(P3, 12, seed=104),
                               f"production {name}"))

    for row in report:
        print(f"  {row['tier']:6s} {row['n_channels']:2d}ch {str(row['hidden']):10s} "
              f"l_p={row['n_predictions']:2d} {row['parameters']:7,d} params  "
              f"crossings {row['crossings']:2d}/{row['steps']}  "
              f"emitted {row['emitted']:2d}/{row['steps']}  "
              f"vec {row['vector_bytes']:,} B")

    # The committed manifest carries only the tiers a fresh clone can regenerate.
    # G4 is built from weights under `runs/`, which is gitignored, so its rows go
    # to a local file and the anti-drift test does not look for them.
    committed = [row for row in report if not row["tier"].startswith("g4")]
    local = [row for row in report if row["tier"].startswith("g4")]
    if committed:
        (VECTORS / "manifest.json").write_text(json.dumps(committed, indent=2) + "\n")
    if local:
        (VECTORS / "manifest_local.json").write_text(json.dumps(local, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
