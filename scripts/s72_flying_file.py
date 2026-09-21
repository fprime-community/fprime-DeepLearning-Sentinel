#!/usr/bin/env python3
"""Section 72 / E5-e, HO1: the flying model file the RETRAINER process loads.

(!) THIS IS NOT THE DEPLOYMENT'S MODEL AND IT IS NOT A SECOND FLOWN MODEL. It is
apparatus. `SentinelRef` flies `SentinelModel.bin` -- 8 channels, hidden [80, 80],
66,960 weights, 268,224 B -- and that file is untouched by everything here.

Why a separate file is needed at all, read from source rather than assumed:

    oxcaml/retrainer/deep_f32.ml:28-34   ins 16, hs 80, n_out 160, n_params 75,360
                                         "ModelFile.hpp:34, at Config.hpp's maxima"
    SentinelModel.bin                    n_channels 8, n_predictions 8
                                         -> 66,960 weights

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

Usage:
    PYTHONPATH=src .venv/bin/python scripts/s72_flying_file.py <out.bin>
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

from sentinel_export import format as fmt
from sentinel_export import writer

#: The retrainer's own shapes. Each is read from `deep_f32.ml`, not chosen here.
N_CHANNELS = 16       # deep_f32.ml:29  ins = 16
HIDDEN = (80, 80)     # deep_f32.ml:28  hs = 80, two layers
N_PREDICTIONS = 10    # deep_f32.ml:31  n_out = 160 = 10 * 16
WINDOW = 250          # deep_f32.ml:32  t_max = 250
DEEP_F32_N_PARAMS = 75360   # deep_f32.ml:34, asserted below

SEED = 0


def build() -> bytes:
    """The flying file, at the retrainer's shapes."""
    shapes = fmt.array_shapes(N_CHANNELS, list(HIDDEN), N_PREDICTIONS, N_CHANNELS)
    total = sum(int(np.prod(shape)) for _, shape in shapes)
    if total != DEEP_F32_N_PARAMS:
        raise SystemExit(
            f"shape drift: this spec yields {total:,} parameters, "
            f"deep_f32.ml:34 declares {DEEP_F32_N_PARAMS:,}. "
            f"One of the two moved; neither may be adjusted to match the other "
            f"without saying so in docs/MODELS.md 72.")

    rng = np.random.default_rng(SEED)
    arrays = {}
    for name, shape in shapes:
        # Small and bounded. Not trained, and never described as trained.
        arrays[name] = rng.uniform(-0.05, 0.05, size=shape).astype(np.float32)

    spec = {
        "arch": "gru",
        "n_channels": N_CHANNELS,
        "n_exogenous": 0,
        "hidden": list(HIDDEN),
        "n_predictions": N_PREDICTIONS,
        "window": WINDOW,
        "channels": [{"id": 1000 + i, "name": f"retrain_{i:02d}"}
                     for i in range(N_CHANNELS)],
        "arrays": arrays,
        "params": {
            "param_version": fmt.PARAM_VERSION_FUSED,
            "norm_policy": "identity",
            "norm_offset": np.zeros(N_CHANNELS, dtype=np.float32),
            "norm_scale": np.ones(N_CHANNELS, dtype=np.float32),
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
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = pathlib.Path(sys.argv[1])
    data = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    weights_bytes = int.from_bytes(data[36:40], "little")
    print(f"   wrote {out} -- {len(data):,} B, "
          f"{N_CHANNELS} channels, hidden {HIDDEN}, "
          f"{weights_bytes // 4:,} float32 weights")
    print(f"   matches deep_f32.ml:34's n_params: "
          f"{weights_bytes // 4 == DEEP_F32_N_PARAMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
