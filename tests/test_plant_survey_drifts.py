"""D85b / docs/MODELS.md 81.9: the premise survey's three healthy drifts, each both ways.

For each drift -- the seasonal eclipse fraction, the housekeeping draw, the instrument's
on-time -- against the same-seed balanced plant:
- before its start tick the drifted plant is the balanced plant, bit for bit;
- after it, the one quantity it names moves as its physics says, near saturation:
  - the eclipse's shadowed fraction of an orbit;
  - the housekeeping draw's floor under the load;
  - the instrument's share of high-load ticks;
- at delta 0 the drifted plant never differs at all.
The default plant's hash is pinned in tests/test_retrainer_learns.py. Needs a C++ compiler.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSIM = ROOT / "fprime" / "SentinelRef" / "PowerSim"

needs_cxx = pytest.mark.skipif(shutil.which("clang++") is None and shutil.which("c++") is None,
                               reason="no C++ compiler")

SRC = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "PowerPlant.hpp"
using namespace Testbed;
int main(int argc, char** argv) {
    const int which = std::atoi(argv[1]);
    const double delta = std::atof(argv[2]);
    const uint32_t spin = 60000u, start = 30000u, end = 200000u;
    PowerPlant h, a; h.reset(2u); a.reset(2u);
    h.configureBalance(0.60, 0.92); a.configureBalance(0.60, 0.92);
    a.configureDrift(static_cast<PowerPlant::Drift>(which), spin + start, 20000.0, delta);
    long firstDiff = -1;
    // Over the last 5 whole orbits: the dark fraction, the minimum load, the high-load share.
    const uint32_t lo = spin + end - 5u * 5400u;
    long darkH = 0, darkA = 0, highH = 0, highA = 0, n = 0;
    double minH = 1e9, minA = 1e9;
    for (uint32_t t = 0; t < spin + end; ++t) {
        h.step(false, 0.0); a.step(false, 0.0);
        for (uint32_t c = 0; c < PLANT_CHANNELS; ++c) {
            const double vh = h.value(c), va = a.value(c);
            if (firstDiff < 0 && std::memcmp(&vh, &va, 8) != 0) firstDiff = (long)t - (long)spin;
        }
        if (t >= lo) {
            ++n;
            darkH += (h.value(CH_SOLAR_INPUT) == 0.0); darkA += (a.value(CH_SOLAR_INPUT) == 0.0);
            highH += (h.value(CH_LOAD_CURRENT) > 3.0); highA += (a.value(CH_LOAD_CURRENT) > 3.0);
            if (h.value(CH_LOAD_CURRENT) < minH) minH = h.value(CH_LOAD_CURRENT);
            if (a.value(CH_LOAD_CURRENT) < minA) minA = a.value(CH_LOAD_CURRENT);
        }
    }
    std::printf("%ld %.6f %.6f %.6f %.6f %.6f %.6f\n", firstDiff, (double)darkH / n,
                (double)darkA / n, (double)highH / n, (double)highA / n, minH, minA);
}
'''


@pytest.fixture(scope="module")
def exe(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("drift")
    cxx = shutil.which("clang++") or shutil.which("c++")
    (d / "drift.cpp").write_text(SRC)
    r = subprocess.run([cxx, "-std=c++14", "-O2", f"-I{POWERSIM}", str(d / "drift.cpp"),
                        str(POWERSIM / "PowerPlant.cpp"), "-o", str(d / "drift")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return d / "drift"


def _run(exe: Path, which: int, delta: float) -> list[float]:
    return [float(x) for x in subprocess.run([str(exe), str(which), str(delta)],
                                             capture_output=True, text=True).stdout.split()]


@needs_cxx
@pytest.mark.parametrize("which,delta", [(0, -0.30), (1, 0.20), (2, -0.30)])
def test_a_drift_moves_nothing_before_its_start(exe, which, delta) -> None:
    first = _run(exe, which, delta)[0]
    assert first >= 30000, f"drift {which} changed the plant at tick {first}, before its start"


@needs_cxx
def test_the_eclipse_drift_shortens_the_shadow(exe) -> None:
    _, dark_h, dark_a, *_ = _run(exe, 0, -0.30)
    assert abs(dark_h - 0.35) < 0.01
    assert abs(dark_a - 0.35 * 0.70) < 0.01, dark_a


@needs_cxx
def test_the_housekeeping_drift_raises_the_load_floor(exe) -> None:
    *_, min_h, min_a = _run(exe, 1, 0.20)
    assert 0.19 < (min_a - min_h) < 0.21, (min_h, min_a)


@needs_cxx
def test_the_duty_drift_shortens_the_instrument_on_time(exe) -> None:
    _, _, _, high_h, high_a, _, _ = _run(exe, 2, -0.30)
    assert abs(high_h - 420 / 900) < 0.01
    assert abs(high_a - 420 * 0.70 / 900) < 0.01, high_a


@needs_cxx
@pytest.mark.parametrize("which", [0, 1, 2])
def test_at_delta_zero_a_drift_changes_nothing(exe, which) -> None:
    assert _run(exe, which, 0.0)[0] == -1
