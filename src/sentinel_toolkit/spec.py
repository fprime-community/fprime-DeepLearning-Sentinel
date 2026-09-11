"""`reference.Weights` -> the dictionary `sentinel_export.write_model` takes.

Moved from `scripts/make_golden_vectors.py:288`, which is where it had to live:
`src/sentinel_export` "depends on nothing else in this repository" by design, so
the bridge from a trained model to the writer's plain dict cannot live inside
it. It could not stay in `scripts/` either, because a shipped entry point under
`src/` cannot import from there except by `importlib`. So it moves here and the
generators import it, which keeps one definition behind every committed vector.

**(!) Three fields that were defaults and are now arguments.** `param_version`,
`tier` and `baseline_only` were hardcoded or omitted at the old call site.
`param_version` omitted is the trap `docs/MODELS.md` 40.6 registers: the writer
defaults it to **1**, so a fused z-score cut written without setting it produces
a version-1 file carrying a version-2 statistic, and both readers accept it.
There is no default here. The caller says which generation it is writing.
"""
from __future__ import annotations

import numpy as np

PROVENANCE_BYTES = 64


def to_spec(weights, threshold: float, warmup: int, provenance: str,
            names: list[str] | None = None, *, param_version: int,
            tier: int = 3, baseline_only: bool = False,
            ewma_span: int = 105) -> dict:
    """The writer's spec. `param_version` is required and is never guessed."""
    n_channels = weights.n_channels
    names = names or [f"channel_{i}" for i in range(n_channels)]
    if len(names) != n_channels:
        raise ValueError(f"{len(names)} names for {n_channels} channels")
    if len(provenance.encode("ascii", "ignore")) >= PROVENANCE_BYTES:
        raise ValueError(
            f"provenance is {len(provenance)} characters and the field holds "
            f"{PROVENANCE_BYTES - 1} plus a NUL; shorten it rather than let the "
            "writer refuse a model that is otherwise fine")
    return {
        "arch": "gru",
        "n_channels": n_channels,
        "n_exogenous": weights.n_exogenous,
        "window": weights.window,
        "n_predictions": weights.n_predictions,
        "hidden": [layer.hidden for layer in weights.layers],
        "channels": [{"id": 1000 + i, "name": names[i]} for i in range(n_channels)],
        "arrays": [(name, np.ascontiguousarray(a, dtype=np.float32))
                   for name, a in weights.arrays()],
        "params": {
            "param_version": int(param_version),
            "ewma_span": int(ewma_span), "agreement": 1, "persistence": 1,
            "warmup_steps": int(warmup),
            "baseline_only": bool(baseline_only), "tier": int(tier),
            "threshold": float(threshold), "provenance": provenance,
            "norm_offset": np.zeros(n_channels, dtype=np.float32),
            "norm_scale": np.ones(n_channels, dtype=np.float32),
        },
    }
