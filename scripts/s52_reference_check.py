#!/usr/bin/env python3
"""Section 52, HD7: two layers and the head against reference.py's OWN forward.

(!) 52.2: reference.py's LAYER forces DTYPE (reference.py:497, :533), unlike the cell 49.3
used, so this is a float32 comparison banded at the project's transcription contract of 1e-5
rather than at machine precision. reference.py is NOT modified -- stop 16 is not triggered.
"""
import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentinel_models import reference  # noqa: E402


def load(path):
    it = iter([t for t in pathlib.Path(path).read_text().split("\n") if t.strip()])
    head = next(it).split()
    assert head[0] == "dims"
    hs, ins, gw, n_out, T = (int(v) for v in head[1:6])
    d = {}
    while True:
        try:
            name, count = next(it).split()
        except StopIteration:
            break
        d[name] = np.array([float(next(it)) for _ in range(int(count))], dtype=np.float64)
    return hs, ins, gw, n_out, T, d


def main() -> int:
    hs, ins, gw, n_out, T, a = load(sys.argv[1] if len(sys.argv) > 1 else "s52_cell.txt")
    f32 = np.float32
    l0 = reference.LayerWeights(
        w_ih=a["w_ih0"].reshape(gw, ins).astype(f32), w_hh=a["w_hh0"].reshape(gw, hs).astype(f32),
        b_ih=a["b_ih0"].astype(f32), b_hh=a["b_hh0"].astype(f32))
    l1 = reference.LayerWeights(
        w_ih=a["w_ih1"].reshape(gw, hs).astype(f32), w_hh=a["w_hh1"].reshape(gw, hs).astype(f32),
        b_ih=a["b_ih1"].astype(f32), b_hh=a["b_hh1"].astype(f32))
    w = reference.Weights(
        layers=(l0, l1), head_w=a["head_w"].reshape(n_out, hs).astype(f32),
        head_b=a["head_b"].astype(f32), n_channels=ins, window=T,
        n_predictions=n_out // ins)
    x = a["x_seq"].reshape(1, T, ins).astype(f32)
    state = [(a["h_init0"].reshape(1, hs).astype(f32),),
             (a["h_init1"].reshape(1, hs).astype(f32),)]
    y_ref, _ = reference.forward(w, x, state, last_only=True)

    print("== Section 52, HD7: two layers and head vs reference.forward ==")
    print(f"   reference.DTYPE {np.dtype(reference.DTYPE).name}; result dtype {y_ref.dtype}")
    diff = np.abs(y_ref.ravel().astype(np.float64) - a["y"])
    worst = float(diff.max()); idx = int(diff.argmax())
    print(f"   worst |OCaml(F64) - reference(F32)| = {worst:.6e}  at index {idx}")
    v = "HOLD" if worst <= 1e-5 else ("NO VERDICT" if worst <= 1e-3 else "FAIL")
    print(f"   band: <=1e-5 HOLD, <=1e-3 NO VERDICT, else FAIL -> {v}")
    return 0 if v == "HOLD" else (1 if v == "NO VERDICT" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
