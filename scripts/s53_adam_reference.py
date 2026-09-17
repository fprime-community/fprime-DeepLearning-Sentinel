#!/usr/bin/env python3
"""Section 53, AD3: the Adam trajectory against a NumPy transcription of 53.2's three details.

(!) 53.2: torch is NOT on the retraining path and is not being added. This transcribes
torch/optim/adam.py as READ -- adam.py:456 (lerp), :475 (mul then addcmul), :530-546 (two
roots divided, eps after). It proves agreement with the algorithm as read and CANNOT prove
agreement with torch as executed; that comparison is registered as owed at 53.2.
"""
import sys
import pathlib
import numpy as np

LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8


def fill(n, scale, off):
    """The OCaml side's LCG, bit for bit: 63-bit ints, so no intermediate overflows."""
    s = 12345 + off
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = ((s * 1103515245) + 12345) & 0x3FFFFFFF
        out[i] = scale * ((float(s % 2000) / 1000.0) - 1.0)
    return out


def main() -> int:
    path = sys.argv[1]
    lines = [ln for ln in pathlib.Path(path).read_text().split("\n") if ln.strip()]
    head = lines[0].split()
    n, steps = int(head[1]), int(head[3])
    got = np.array([float(x) for x in lines[1:1 + n]], dtype=np.float64)

    p = fill(n, 0.5, 31)
    m = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    for t in range(1, steps + 1):
        g = fill(n, 1.0, 1000 + t)
        bc1 = 1.0 - B1 ** float(t)          # adam.py:530
        bc2 = 1.0 - B2 ** float(t)          # adam.py:531
        ss = LR / bc1                       # adam.py:533
        b2s = bc2 ** 0.5                    # adam.py:535
        m = m + ((1.0 - B1) * (g - m))      # adam.py:456, the lerp
        v = (v * B2) + ((1.0 - B2) * g * g)  # adam.py:475, NOT a lerp
        denom = (np.sqrt(v) / b2s) + EPS    # adam.py:544, two roots divided
        p = p + ((-ss) * (m / denom))       # adam.py:546, addcdiv with -step_size

    print("== Section 53, AD3: Adam trajectory vs torch's algorithm as read ==")
    print(f"   {steps} updates over {n} parameters")
    denom_ref = np.maximum(np.abs(p), 1e-300)
    rel = np.abs(got - p) / denom_ref
    worst = float(rel.max()); idx = int(rel.argmax())
    print(f"   worst relative difference {worst:.6e} at index {idx}")
    print(f"   worst absolute difference {float(np.abs(got - p).max()):.6e}")
    v_ = "HOLD" if worst <= 1e-12 else ("NO VERDICT" if worst <= 1e-9 else "FAIL")
    print(f"   band: <=1e-12 HOLD, <=1e-9 NO VERDICT, else FAIL -> {v_}")
    return 0 if v_ == "HOLD" else (1 if v_ == "NO VERDICT" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
