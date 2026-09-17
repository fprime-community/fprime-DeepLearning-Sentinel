#!/usr/bin/env python3
"""Section 63, OW1-OW3: torch's Adam as EXECUTED, and the association 53.9 left open.

(!) 53.2 and stop 36: torch is NOT on the retraining path and is not being added. This
reads it as a reference and that is the only thing it may ever be.

53.9 registered two owed items. AD3 proved agreement with the algorithm as READ at
`torch/optim/adam.py`; it could not prove agreement with what torch RUNS, which dispatches
between `_single_tensor_adam` (:347), `_multi_tensor_adam` (:553) and `_fused_adam` (:802)
at :937-968. And `adam.py:475` is

    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

which computes `value * t1 * t2` WITHOUT the Python saying how it associates. Both the
NumPy reference (`scripts/s53_adam_reference.py:42`) and the OxCaml side
(`oxcaml/retrainer/adam.ml:72-73`) write `((1-b2)*g)*g`, because `*` and `*.` are both
left-associative -- so both MIRROR a choice neither verified.

The gradient sequence and the initial parameters come from the reference module itself,
imported rather than re-implemented, so bit-identity is guaranteed and not asserted.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
N, STEPS = 75360, 20          # 53's model at ModelFile.hpp:34's maxima; AD3's 20 updates


def _reference_module():
    """`scripts/s53_adam_reference.py`, imported for its LCG. Never re-implemented."""
    path = ROOT / "scripts" / "s53_adam_reference.py"
    spec = importlib.util.spec_from_file_location("s53_adam_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def numpy_reference(fill, n, steps):
    """`scripts/s53_adam_reference.py`'s trajectory, the algorithm AS READ."""
    p = fill(n, 0.5, 31)
    m = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    for t in range(1, steps + 1):
        g = fill(n, 1.0, 1000 + t)
        bc1 = 1.0 - B1 ** float(t)
        bc2 = 1.0 - B2 ** float(t)
        ss = LR / bc1
        b2s = bc2 ** 0.5
        m = m + ((1.0 - B1) * (g - m))
        v = (v * B2) + ((1.0 - B2) * g * g)
        denom = (np.sqrt(v) / b2s) + EPS
        p = p + ((-ss) * (m / denom))
    return p


def torch_trajectory(fill, n, steps, **adam_kwargs):
    """torch's Adam AS EXECUTED, float64 on CPU, gradients set by hand."""
    p0 = fill(n, 0.5, 31)
    param = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([param], lr=LR, betas=(B1, B2), eps=EPS, **adam_kwargs)
    for t in range(1, steps + 1):
        g = fill(n, 1.0, 1000 + t)
        param.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
    return param.detach().numpy().copy(), opt


def ow1_association():
    """OW1: which way does `addcmul_` associate? Probed where the two answers differ.

    `exp_avg_sq` starts at zero, so `.mul_(beta2)` leaves zero and `.addcmul_` alone
    produces the product -- which isolates the association exactly.
    """
    value = 1.0 - B2
    rng = np.random.default_rng(63)
    probes = []
    while len(probes) < 12:
        g = float(rng.uniform(-3.0, 3.0))
        left = (value * g) * g            # ((1-b2)*g)*g -- what both sides mirror
        right = value * (g * g)           # (1-b2)*(g*g)
        if left != right:                 # a separating probe
            probes.append((g, left, right))

    print("-- OW1: the addcmul_ association, on separating probes")
    print(f"   value = 1 - beta2 = {value!r}")
    print(f"   {len(probes)} probes where ((1-b2)*g)*g != (1-b2)*(g*g)")
    matches_left = matches_right = matches_neither = 0
    for g, left, right in probes:
        v = torch.zeros(1, dtype=torch.float64)
        got = float(v.mul_(B2).addcmul_(
            torch.tensor([g], dtype=torch.float64),
            torch.tensor([g], dtype=torch.float64),
            value=value)[0])
        if got == left:
            matches_left += 1
        elif got == right:
            matches_right += 1
        else:
            matches_neither += 1
    print(f"   bit-identical to ((1-b2)*g)*g : {matches_left} of {len(probes)}")
    print(f"   bit-identical to (1-b2)*(g*g) : {matches_right} of {len(probes)}")
    print(f"   neither                       : {matches_neither} of {len(probes)}")
    if not probes:
        return "NO VERDICT", "no probe separated the two associations"
    if matches_left == len(probes):
        return "HOLD", "torch associates left, which is what both sides already write"
    if matches_right > 0:
        return "FAIL", "torch associates right; the transcription mirrored the wrong one"
    return "FAIL", "torch matches neither association"


def _resolved_path():
    """Which of adam.py:964-968's three funcs the DEFAULT settings select on this host.

    `opt.param_groups[0]` stores the UNRESOLVED `None`, so reading it back names nothing.
    The resolution is `_default_to_fused_or_foreach` at `adam.py:938-940`, and it is
    asked here directly rather than inferred.
    """
    from torch.optim.optimizer import _default_to_fused_or_foreach
    probe = torch.zeros(1, dtype=torch.float64)
    fused, foreach = _default_to_fused_or_foreach(
        [probe], differentiable=False, use_fused=False)
    if fused:
        return "_fused_adam (adam.py:802)"
    if foreach:
        return "_multi_tensor_adam (adam.py:553)"
    return "_single_tensor_adam (adam.py:347)"


def _band(worst):
    return "HOLD" if worst <= 1e-12 else ("NO VERDICT" if worst <= 1e-9 else "FAIL")


def main() -> int:
    mod = _reference_module()
    fill = mod.fill
    print("== Section 63, OW1-OW3: torch's Adam as executed ==")
    print(f"   torch {torch.__version__}, float64, CPU; {STEPS} updates over {N:,} parameters")
    print()

    ow1, ow1_why = ow1_association()
    print(f"   OW1 -> {ow1}  ({ow1_why})")
    print()

    ref = numpy_reference(fill, N, STEPS)

    print("-- OW2: torch's DEFAULT path against the algorithm as read")
    got, opt = torch_trajectory(fill, N, STEPS)
    print(f"   the path torch actually resolved to: {_resolved_path()}")
    denom = np.maximum(np.abs(ref), 1e-300)
    rel = np.abs(got - ref) / denom
    worst2 = float(rel.max())
    print(f"   worst relative difference {worst2:.6e} at index {int(rel.argmax())}")
    print(f"   worst absolute difference {float(np.abs(got - ref).max()):.6e}")
    ow2 = _band(worst2)
    print(f"   band: <=1e-12 HOLD, <=1e-9 NO VERDICT, else FAIL -> {ow2}")
    print()

    print("-- OW3: torch's own paths against each other")
    single, _ = torch_trajectory(fill, N, STEPS, foreach=False)
    multi, _ = torch_trajectory(fill, N, STEPS, foreach=True)
    identical = bool(np.array_equal(single, multi))
    diff = float(np.abs(single - multi).max())
    print(f"   _single_tensor_adam vs _multi_tensor_adam: "
          f"{'BIT-IDENTICAL' if identical else 'DIFFER'}, max abs {diff:.6e}")
    s_rel = float((np.abs(single - ref) / denom).max())
    m_rel = float((np.abs(multi - ref) / denom).max())
    print(f"   single vs reference: worst relative {s_rel:.6e} -> {_band(s_rel)}")
    print(f"   multi  vs reference: worst relative {m_rel:.6e} -> {_band(m_rel)}")
    print("   _fused_adam: not exercised -- CUDA-only on this host, reported as unavailable")
    ow3 = "HOLD" if identical else "FAIL"
    print(f"   OW3 -> {ow3}")
    print()

    print(f"   OW1 {ow1} | OW2 {ow2} | OW3 {ow3}")
    return 0 if all(v == "HOLD" for v in (ow1, ow2, ow3)) else 1


if __name__ == "__main__":
    sys.exit(main())
