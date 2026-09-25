#!/usr/bin/env python3
"""Section 72 / E5-e, HO1: the flying model file the RETRAINER process loads.

(!) THIS IS NOT THE DEPLOYMENT'S MODEL AND IT IS NOT A SECOND FLOWN MODEL. It is
apparatus. `SentinelRef` flies `SentinelModel.bin` -- 8 channels, hidden [80, 80],
66,960 weights, 268,224 B -- and that file is untouched by everything here.

Why a separate file is needed at all, read from source rather than assumed:

    oxcaml/retrainer/deep_f32.ml:35-41   ins 16, hs 80, n_out 160, n_params 75,360
                                         "ModelFile.hpp:34, at Config.hpp's maxima"
    SentinelModel.bin                    n_channels 8, n_predictions 10
                                         -> 66,960 weights

(The line above said "n_predictions 8" until 2026-09-25. 8 x 8 gives 65,664, not
66,960; `docs/MODELS.md` 72.10 re-derived it as 8 channels x 10 predictions.)

`Deep_f32` is a FIXED, MAXIMA-SHAPED network. Its cycle exports 75,360 float32.
`shadow59.ml:79` refuses a weights block whose size disagrees with the header
(`if wb <> 4 * n_weights then (-1)`), and it is right to: writing 75,360 numbers
from one architecture into a container declaring 66,960 would produce a file that
loads and means nothing.

So the retrainer's flying file is the one shape its cycle actually trains --
16 channels, hidden [80, 80], 10 predictions, exactly 75,360 parameters, verified
against `Deep_f32.n_params` below rather than trusted.

(!) AND THE GAP THIS EXPOSES IS REPORTED, NOT PAPERED OVER. 59.2's premise is
that "the retrainer trains the same architecture at the same shapes" as the
flying model. For this deployment it does not, and no candidate the retrainer
builds can be loaded by `SentinelRef` as a replacement for `SentinelModel.bin`.
docs/MODELS.md 72 records that as a finding and 72.6 carries it as owed.

The weights are deterministic from a seed and are NOT trained. They exist so the
crossing has a well-formed container to overwrite; the candidate's weights come
from the cycle, which is the thing under test.

(!) D84: AND AT A MISSION'S SHAPE. `scripts/oxcaml_shape.py` generates `deep_f32`
at a mission's channel and prediction count (D83.1 route 1), and HO1 must be re-run
against a flying file at THAT shape. `--channels` and `--predictions` build one;
with neither given the output is byte-identical to what this script wrote before,
which tests/test_retrainer_shape_is_generated.py pins by hash.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/s72_flying_file.py <out.bin>
    PYTHONPATH=src .venv/bin/python scripts/s72_flying_file.py <out.bin> \
        --channels 8 --predictions 10
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

from sentinel_export import format as fmt
from sentinel_export import writer

#: The retrainer's own shapes. Each is read from `deep_f32.ml`, not chosen here.
N_CHANNELS = 16       # deep_f32.ml:36  ins = 16
HIDDEN = (80, 80)     # deep_f32.ml:35  hs = 80, two layers
N_PREDICTIONS = 10    # deep_f32.ml:38  n_out = 160 = 10 * 16
WINDOW = 250          # deep_f32.ml:39  t_max = 250
DEEP_F32_N_PARAMS = 75360   # deep_f32.ml:41, asserted below

SEED = 0


def build(n_channels: int = N_CHANNELS, n_predictions: int = N_PREDICTIONS) -> bytes:
    """The flying file, at the retrainer's shapes -- the maxima unless told otherwise."""
    shapes = fmt.array_shapes(n_channels, list(HIDDEN), n_predictions, n_channels)
    total = sum(int(np.prod(shape)) for _, shape in shapes)
    # At the maxima the reference is deep_f32.ml's own literal; at a mission shape
    # it is the generator's count, which is what the generated deep_f32 declares.
    if (n_channels, n_predictions) == (N_CHANNELS, N_PREDICTIONS):
        expected, source = DEEP_F32_N_PARAMS, "deep_f32.ml:41"
    else:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import oxcaml_shape
        try:
            oxcaml_shape.check_box(n_channels, n_predictions)
        except oxcaml_shape.ShapeError as exc:
            raise SystemExit(f"s72_flying_file: refused -- {exc}") from None
        expected = oxcaml_shape.parameter_count(n_channels, n_predictions)
        source = "scripts/oxcaml_shape.py"
    if total != expected:
        raise SystemExit(
            f"shape drift: this spec yields {total:,} parameters, "
            f"{source} declares {expected:,}. "
            f"One of the two moved; neither may be adjusted to match the other "
            f"without saying so in docs/MODELS.md 72.")

    rng = np.random.default_rng(SEED)
    arrays = {}
    for name, shape in shapes:
        # Small and bounded. Not trained, and never described as trained.
        arrays[name] = rng.uniform(-0.05, 0.05, size=shape).astype(np.float32)

    spec = {
        "arch": "gru",
        "n_channels": n_channels,
        "n_exogenous": 0,
        "hidden": list(HIDDEN),
        "n_predictions": n_predictions,
        "window": WINDOW,
        "channels": [{"id": 1000 + i, "name": f"retrain_{i:02d}"}
                     for i in range(n_channels)],
        "arrays": arrays,
        "params": {
            "param_version": fmt.PARAM_VERSION_FUSED,
            "norm_policy": "identity",
            "norm_offset": np.zeros(n_channels, dtype=np.float32),
            "norm_scale": np.ones(n_channels, dtype=np.float32),
            "ewma_span": 105,
            "agreement": 1,
            "persistence": 1,
            "warmup_steps": 2350,
            "baseline_only": False,
            "tier": 3,
            "threshold": 1.0,
            "provenance": "s72 apparatus, seeded, not trained",
        },
    }
    return writer.write_model(spec)


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__)
        return 2
    out = pathlib.Path(args[0])
    shape = {"--channels": N_CHANNELS, "--predictions": N_PREDICTIONS}
    rest = args[1:]
    while rest:
        if len(rest) < 2 or rest[0] not in shape:
            print(__doc__)
            return 2
        shape[rest[0]] = int(rest[1])
        rest = rest[2:]
    n_channels, n_predictions = shape["--channels"], shape["--predictions"]
    data = build(n_channels, n_predictions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    weights_bytes = int.from_bytes(data[36:40], "little")
    print(f"   wrote {out} -- {len(data):,} B, "
          f"{n_channels} channels, hidden {HIDDEN}, "
          f"{weights_bytes // 4:,} float32 weights")
    if (n_channels, n_predictions) == (N_CHANNELS, N_PREDICTIONS):
        print(f"   matches deep_f32.ml:41's n_params: "
              f"{weights_bytes // 4 == DEEP_F32_N_PARAMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
