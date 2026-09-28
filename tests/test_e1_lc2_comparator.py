"""docs/MODELS.md 78.11, LC2: the comparator and the rung conditions, both directions.

`scripts/d85_lc2.py` reads the two deployments' own logs by tap sequence number. These
tests feed it synthetic log text in the exact line formats the EmitProbe and the
Retrainer write, and show that each verdict can go both ways: identical streams HOLD,
one shifted tick FAILS and is named, two empty streams FAIL (vacuity), and each
validity condition turns the verdict into NO VERDICT.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d85_lc2", ROOT / "scripts" / "d85_lc2.py")
lc2 = importlib.util.module_from_spec(_spec)
sys.modules["d85_lc2"] = lc2      # a dataclass resolves its module by name
_spec.loader.exec_module(lc2)


def _ref(emits, last=3000, extra=""):
    lines = ["PROBE_FIRST seq 0", "PROBE_MODE seq 0 model 1"]
    lines += [f"TICK seq {s}" for s in range(0, last + 1, 1000)]
    lines += [f"EMIT seq {s}" for s in emits]
    return "\n".join(lines) + "\n" + extra


def _rep(emits, last=3000, extra=""):
    lines = ["REPLICA_FIRST seq 0"]
    lines += [f"REPLICA_TICK seq {s}" for s in range(0, last + 1, 1000)]
    lines += [f"REPLICA_EMIT seq {s}" for s in emits]
    return "\n".join(lines) + "\n" + extra


def _verdict(ref_text, rep_text, rg=0):
    ref = lc2.read_side(ref_text, replica=False)
    rep = lc2.read_side(rep_text, replica=True)
    res = lc2.compare(ref.emits, rep.emits, min(ref.last, rep.last))
    return lc2.validity(ref, rep, rg), res


def test_identical_streams_hold() -> None:
    why, res = _verdict(_ref([2400, 2401, 2900]), _rep([2400, 2401, 2900]))
    assert why == [] and res["verdict"] == "HOLD" and res["n_monitor"] == 3


def test_one_shifted_tick_fails_and_is_named() -> None:
    why, res = _verdict(_ref([2400, 2900]), _rep([2400, 2901]))
    assert why == [] and res["verdict"] == "FAIL"
    assert res["only_monitor"] == [2900] and res["only_replica"] == [2901]


def test_two_empty_streams_fail() -> None:
    _, res = _verdict(_ref([]), _rep([]))
    assert res["verdict"] == "FAIL" and "vacuous" in res["why"]


def test_emits_past_the_common_range_are_not_compared() -> None:
    # The replica reached 3,000, the probe 2,000: an emit at 2,500 on one side only is
    # beyond what both sides processed, and is not a difference.
    _, res = _verdict(_ref([1500], last=2000), _rep([1500, 2500], last=3000))
    assert res["hi"] == 2000 and res["verdict"] == "HOLD"


def test_each_validity_condition_withholds_the_verdict() -> None:
    good_ref, good_rep = _ref([2400]), _rep([2400])
    assert _verdict(good_ref, good_rep)[0] == []
    cases = {
        "first sequence": (good_ref, _rep([2400]).replace("REPLICA_FIRST seq 0", "REPLICA_FIRST seq 3")),
        "lost": (good_ref, good_rep + "EVENT: SamplesLost 3\n"),
        "gap": (good_ref, good_rep + "REPLICA_GAP seq 1500 after 1497 filled 3\n"),
        "slip": (good_ref + "WARNING_HI: rateGroup_1Hz.RateGroupCycleSlip 77\n", good_rep),
        "mode": (good_ref + "PROBE_MODE seq 1800 model 0\n", good_rep),
        "reload": (good_ref + "sentinelMonitor.ModelReloadAccepted RetrainApproved.bin\n", good_rep),
        "stale": (good_ref + "PROBE_STALE seq 1200\n", good_rep),
    }
    for name, (r, p) in cases.items():
        assert _verdict(r, p)[0] != [], name
    assert _verdict(good_ref, good_rep, rg=2)[0] != [], "RgCycleSlips"


def test_the_rung_conditions_both_ways() -> None:
    ref, rep = lc2.read_side(_ref([]), False), lc2.read_side(_rep([]), True)
    assert all(ok for _, ok, _ in lc2.calibrate(ref, rep, 0))
    lost = lc2.read_side(_rep([]) + "SamplesLost 4\n", True)
    assert not all(ok for _, ok, _ in lc2.calibrate(ref, lost, 0))
    assert not all(ok for _, ok, _ in lc2.calibrate(ref, rep, 1))


def test_rg_slips_reads_the_largest_value() -> None:
    log = ("2026 SentinelRef.rateGroup_1Hz.RgCycleSlips 1\n"
           "2026 SentinelRef.rateGroup_1Hz.RgMaxTime 900\n"
           "2026 SentinelRef.rateGroup_1Hz.RgCycleSlips 3\n")
    assert lc2.rg_slips(log) == 3 and lc2.rg_slips("") == 0
