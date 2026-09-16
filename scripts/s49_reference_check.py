#!/usr/bin/env python3
"""Section 49, Arm B (J4): the OxCaml cell's forward against reference.py's own, at float64.

(!) reference.py is NOT MODIFIED, and 49.3 records why it does not need to be. DTYPE there
is np.float32, but it is applied at the LAYER entry points; `gru_cell` never mentions it and
`_sigmoid` allocates with np.empty_like, so float64 arrays propagate float64 end to end.
Stop 16 fires if this file ever needs to change that.
"""
import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentinel_models import reference  # noqa: E402


def load(path):
    toks = pathlib.Path(path).read_text().split("\n")
    it = iter([t for t in toks if t.strip()])
    head = next(it).split()
    assert head[0] == "dims"
    hs, ins, gw = (int(v) for v in head[1:4])
    out = {}
    for _ in range(7):
        name, count = next(it).split()
        out[name] = np.array([float(next(it)) for _ in range(int(count))], dtype=np.float64)
    return hs, ins, gw, out


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "s49_cell.txt"
    hs, ins, gw, a = load(path)

    # reference.py is float32 BY DECLARATION (reference.py:68) and float64 BY PROPAGATION
    # when gru_cell is called directly. Assert the dtype actually survives, rather than
    # trusting the reading -- 49.3's whole argument is this one fact.
    w_ih = a["w_ih"].reshape(gw, ins)
    w_hh = a["w_hh"].reshape(gw, hs)
    x = a["x"].reshape(1, ins)
    h = a["h"].reshape(1, hs)
    projected = x @ w_ih.T + a["b_ih"]
    (h_ref,) = reference.gru_cell(projected, h, w_hh, a["b_hh"])

    print("== Section 49, Arm B: OxCaml cell vs reference.py's gru_cell, float64 ==")
    print(f"   reference.DTYPE is {np.dtype(reference.DTYPE).name} by declaration")
    print(f"   projected dtype {projected.dtype}, result dtype {h_ref.dtype}")
    if h_ref.dtype != np.float64:
        print("   (!) STOP: float64 did not propagate. 49.3's premise is wrong.")
        return 3

    diff = np.abs(h_ref.ravel() - a["h_new"])
    worst = float(diff.max())
    idx = int(diff.argmax())
    print(f"   worst |OCaml(F64) - reference(F64)| = {worst:.6e}  at index {idx}")
    if worst <= 1e-12:
        verdict = "HOLD"
    elif worst <= 1e-9:
        verdict = "NO VERDICT"
    else:
        verdict = "FAIL"
    print(f"   band: <=1e-12 HOLD, <=1e-9 NO VERDICT, else FAIL -> {verdict}")
    return 0 if verdict == "HOLD" else (1 if verdict == "NO VERDICT" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
