"""docs/MODELS.md 78.11, HT1-HT4: the retrainer's window fills from live samples, and no
tick within SPAN + 2 x guard of a crossing is admitted -- through the Retrainer component.

`Sentinel_Retrainer_ut_exe --gtest_filter=Retrainer.LiveWindow` drives the real
component through its real ports -- `sampleIn`, as the hub's thread would, then
`schedIn`, as the rate group would -- with SentinelRetrain's own loop constants and the
committed test model (`SENTINEL_RETRAINER_UT_FLYING`). It writes one row per tap
sequence: the `Admitted` telemetry value, the replica's crossing as the loop gates it,
its emit, and 56's `admits`. Two processes, because OCaml's state is process-global:
a CONTROL run of the quiet input, and a SPIKE run with one in-band spike at f = 7,000.

    HT1  both runs: nothing admitted before 56's window is full (sequence 6,549), then
         one admission on every sequence of [6,549, 7,000), with nothing flagged
    HT2  spike run: the replica crosses at f, and no admitted sequence lies within 780
         (SPAN 260 + 2 x guard 260) of the last flagged sequence before it
    HT3  control run: every sequence of [f, f + 780) IS admitted, so the refusal is the
         crossing's
    HT4  spike run: after the burst, admission resumes where the band has cleared and
         56 admits -- not before, and not later

Each checker is also shown failing on a doctored trace (the other direction).
(!) No duration is measured or recorded (stop 35).
"""
from __future__ import annotations

import csv
import hashlib
import os
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UT_CACHE = ROOT / "fprime" / "build-fprime-automatic-native-ut"
RETRAINER_UT = UT_CACHE / "bin" / "Darwin" / "Sentinel_Retrainer_ut_exe"
#: 78.11's test model: a byte-copy of runs/d85/flying/flying.bin (8 channels, 10
#: predictions, 268,224 B -- the smallest file the 8/10 build can load).
FLYING = ROOT / "flight" / "test" / "vectors" / "retrainer_ut_flying_c8_p10.bin"
FLYING_SHA256 = "8b22e32159455d03bccfc7c34fa2236a0bca69ddab32aa309b70326a3eb461a9"

# 78.11's constants.
WINDOW_FULL = 6549          # the 6,550th push: 56's window is full
SPIKE = 7000                # f
EXTENT = 780                # SPAN 260 + 2 x guard 260
CEILING = 14                # 56's emissions ceiling at 1,046 ppm


def _cache_value(name: str) -> str | None:
    cache = UT_CACHE / "CMakeCache.txt"
    if not cache.exists():
        return None
    m = re.search(rf"^{name}:\w+=(\S+)$", cache.read_text(encoding="utf-8", errors="replace"),
                  re.M)
    return m.group(1) if m else None


def _run(out: pathlib.Path, spike: bool) -> list[dict]:
    trace = out / ("spike.csv" if spike else "control.csv")
    env = dict(os.environ, SENTINEL_RETRAINER_UT_LIVE="1",
               SENTINEL_RETRAINER_UT_FLYING=str(FLYING),
               SENTINEL_RETRAINER_UT_TRACE=str(trace),
               SENTINEL_RETRAINER_UT_CANDIDATE=str(out / "candidate.bin"),
               SENTINEL_RETRAINER_UT_SPIKE="1" if spike else "0")
    r = subprocess.run([str(RETRAINER_UT), "--gtest_filter=Retrainer.LiveWindow"],
                       capture_output=True, text=True, env=env, cwd=str(out), timeout=3600)
    text = r.stdout + r.stderr
    assert r.returncode == 0, text[-3000:]
    assert "[  SKIPPED ]" not in text, "LiveWindow skipped: the runner did not set it up"
    assert "[  PASSED  ] 1 test" in text, text[-3000:]
    keep = os.environ.get("SENTINEL_LIVE_WINDOW_KEEP")
    if keep:
        pathlib.Path(keep).mkdir(parents=True, exist_ok=True)
        shutil.copy(trace, pathlib.Path(keep) / trace.name)
    with trace.open() as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    if (_cache_value("SENTINEL_WITH_RETRAINER") or "").upper() not in ("ON", "TRUE", "1"):
        pytest.skip("the unit-test cache is not configured with SENTINEL_WITH_RETRAINER=ON")
    if (_cache_value("SENTINEL_RETRAINER_CHANNELS"), _cache_value("SENTINEL_RETRAINER_PREDICTIONS")) \
            != ("8", "10"):
        pytest.skip("the unit-test cache is not built at 8 channels and 10 predictions")
    if not RETRAINER_UT.exists():
        pytest.skip(f"{RETRAINER_UT.name} is not built; run fprime-util check in "
                    "fprime/Sentinel/Retrainer")
    got = hashlib.sha256(FLYING.read_bytes()).hexdigest()
    assert got == FLYING_SHA256, f"{FLYING.name} is {got}, not 78.11's model"
    out = tmp_path_factory.mktemp("live_window")
    return {"control": _run(out, False), "spike": _run(out, True)}


# -- the checkers, each a list of failures ------------------------------------------------

def _admitted(rows: list[dict]) -> list[bool]:
    """Per sequence: did the Admitted channel rise on it."""
    prev, out = 0, []
    for r in rows:
        out.append(r["admitted"] > prev)
        prev = r["admitted"]
    return out


def _flagged(r: dict) -> bool:
    return bool(r["crossing"] or r["emitted"])


def check_ht1(rows: list[dict]) -> list[str]:
    bad = []
    adm = _admitted(rows)
    early = [i for i in range(WINDOW_FULL) if adm[i]]
    if early:
        bad.append(f"admitted before the window was full: {early[:5]}")
    missed = [i for i in range(WINDOW_FULL, SPIKE) if not adm[i]]
    if missed:
        bad.append(f"quiet sequences not admitted: {missed[:5]} ({len(missed)})")
    flags = [i for i in range(SPIKE) if _flagged(rows[i])]
    if flags:
        bad.append(f"the quiet stretch was flagged at {flags[:5]}")
    return bad


def check_ht2(rows: list[dict]) -> list[str]:
    bad = []
    if not rows[SPIKE]["crossing"]:
        bad.append(f"the replica did not cross at {SPIKE}")
    adm = _admitted(rows)
    last = None
    for i, r in enumerate(rows):
        if _flagged(r):
            last = i
        if adm[i] and last is not None and i - last < EXTENT:
            bad.append(f"sequence {i} admitted {i - last} after the flag at {last}")
    return bad


def check_ht3(rows: list[dict]) -> list[str]:
    adm = _admitted(rows)
    missed = [i for i in range(SPIKE, SPIKE + EXTENT) if not adm[i]]
    return [f"control: band sequences not admitted: {missed[:5]} ({len(missed)})"] if missed else []


def burst_end(rows: list[dict]) -> int:
    return max(i for i, r in enumerate(rows) if i >= SPIKE and _flagged(r))


def check_ht4(rows: list[dict]) -> list[str]:
    bad = []
    adm = _admitted(rows)
    end = burst_end(rows)
    for i in range(end + 1, len(rows)):
        cleared = i - end >= EXTENT
        if adm[i] and not cleared:
            bad.append(f"admitted at {i}, before the band cleared at {end + EXTENT}")
        if cleared and rows[i]["window_admits"] and not adm[i]:
            bad.append(f"not admitted at {i}: the band had cleared and 56 admitted")
    return bad


def summary(rows: list[dict]) -> dict:
    """What 78.12 records, in sequence numbers only."""
    adm = _admitted(rows)
    end = burst_end(rows)
    after = [i for i in range(end + 1, len(rows)) if adm[i]]
    return {"burst_end": end,
            "burst_emits": sum(r["emitted"] for r in rows[SPIKE:]),
            "first_admission_after": after[0] if after else None,
            "admitted_total": rows[-1]["admitted"]}


# -- the tests ------------------------------------------------------------------------------

def test_ht1_the_window_fills_from_live_samples(traces) -> None:
    for name in ("control", "spike"):
        assert check_ht1(traces[name]) == [], name


def test_ht2_nothing_within_the_band_of_a_crossing_is_admitted(traces) -> None:
    assert check_ht2(traces["spike"]) == []


def test_ht3_the_same_ticks_are_admitted_without_the_crossing(traces) -> None:
    assert check_ht3(traces["control"]) == []
    assert check_ht2(traces["control"]) != [], "the control must not cross at f"


def test_ht4_the_band_ends_where_it_should(traces) -> None:
    rows = traces["spike"]
    assert check_ht4(rows) == []
    s = summary(rows)
    print(f"HT4: burst ends at {s['burst_end']}, {s['burst_emits']} emits after f "
          f"(ceiling {CEILING}); first admission after it: {s['first_admission_after']}")


# -- the other direction: each checker catches a doctored trace ----------------------------

def _quiet(n: int = SPIKE + EXTENT + 400) -> list[dict]:
    rows, total = [], 0
    for i in range(n):
        total += 1 if i >= WINDOW_FULL else 0
        rows.append({"seq": i, "admitted": total, "crossing": 0, "emitted": 0,
                     "window_admits": 1 if i >= WINDOW_FULL else 0})
    return rows


def _with_spike(rows: list[dict], admit_inside: int | None = None) -> list[dict]:
    rows = [dict(r) for r in rows]
    rows[SPIKE]["crossing"] = rows[SPIKE]["emitted"] = 1
    total = rows[SPIKE - 1]["admitted"]
    for i in range(SPIKE, len(rows)):
        if i - SPIKE >= EXTENT or i == admit_inside:
            total += 1
        rows[i]["admitted"] = total
    return rows


def test_the_checkers_pass_a_correct_trace() -> None:
    q = _quiet()
    s = _with_spike(q)
    assert check_ht1(q) == [] and check_ht1(s) == []
    assert check_ht2(s) == [] and check_ht3(q) == [] and check_ht4(s) == []


def test_ht2_catches_an_admission_inside_the_band() -> None:
    assert check_ht2(_with_spike(_quiet(), admit_inside=SPIKE + 100)) != []


def test_ht4_catches_an_early_admission() -> None:
    assert check_ht4(_with_spike(_quiet(), admit_inside=SPIKE + EXTENT - 1)) != []


def test_ht1_catches_an_unadmitted_quiet_stretch() -> None:
    q = _quiet()
    for r in q:
        r["admitted"] = 0
    assert check_ht1(q) != []


def test_ht1_catches_an_admission_before_the_window_is_full() -> None:
    q = _quiet()
    for r in q[100:]:
        r["admitted"] += 1
    assert check_ht1(q) != []


def test_ht3_catches_a_control_that_refuses_the_band() -> None:
    q = _quiet()
    for r in q[SPIKE + 10:]:
        r["admitted"] -= 1
    assert check_ht3(q) != []
