"""D85b / docs/MODELS.md 81.3: arm (a)'s battery open-circuit-voltage ageing, both ways.

- Before its start tick the aged plant is the balanced plant, bit for bit; unconfigured, the
  default plant's hash is untouched (tests/test_retrainer_learns.py pins it).
- After it, the bus sits lower by about 4.5 x L x g x SoC, which is the physics 81.3 names,
  and at the ladder's largest rung (L = 0.06) every channel stays inside yellow.
- The other direction: at L = 0 the "aged" plant never differs from the balanced one, so the
  change is the ageing and nothing else.
Needs a C++ compiler.
"""
from __future__ import annotations

import math
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
static const double YLO[8] = {-2.0,-10.0,0.5,27.0,-5.0,-35.0,0.0,0.30};
static const double YHI[8] = {300.0,10.0,9.5,32.0,40.0,30.0,1.0,1.0};
int main(int argc, char** argv) {
    const double loss = std::atof(argv[1]);
    const uint32_t spin = 60000u, start = 30000u, end = 130000u;
    PowerPlant h, a; h.reset(1u); a.reset(1u);
    h.configureBalance(0.60, 0.92); a.configureBalance(0.60, 0.92);
    if (loss >= 0.0) a.configureOcvAgeing(spin + start, 20000.0, loss);
    long firstDiff = -1, yellow = -1;
    double busH = 0.0, busA = 0.0, soc = 0.0;
    for (uint32_t t = 0; t < spin + end; ++t) {
        h.step(false, 0.0); a.step(false, 0.0);
        for (uint32_t c = 0; c < PLANT_CHANNELS; ++c) {
            const double vh = h.value(c), va = a.value(c);
            if (firstDiff < 0 && std::memcmp(&vh, &va, 8) != 0) firstDiff = (long)t - (long)spin;
            if (t >= spin && yellow < 0 && (va < YLO[c] || va > YHI[c])) yellow = (long)t - (long)spin;
        }
        if (t == spin + end - 1u) {
            busH = h.value(CH_BUS_VOLTAGE); busA = a.value(CH_BUS_VOLTAGE);
            soc = a.value(CH_STATE_OF_CHARGE);
        }
    }
    std::printf("%ld %ld %.9f %.9f\n", firstDiff, yellow, busA - busH, soc);
}
'''


def _run(tmp_path: Path, loss: float) -> tuple[int, int, float, float]:
    cxx = shutil.which("clang++") or shutil.which("c++")
    exe = tmp_path / "ocv"
    if not exe.exists():
        (tmp_path / "ocv.cpp").write_text(SRC)
        r = subprocess.run([cxx, "-std=c++14", "-O2", f"-I{POWERSIM}", str(tmp_path / "ocv.cpp"),
                            str(POWERSIM / "PowerPlant.cpp"), "-o", str(exe)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    first, yellow, dbus, soc = subprocess.run([str(exe), str(loss)], capture_output=True,
                                              text=True).stdout.split()
    return int(first), int(yellow), float(dbus), float(soc)


@needs_cxx
def test_ocv_ageing_moves_the_bus_after_its_start_and_never_before(tmp_path: Path) -> None:
    first, yellow, dbus, soc = _run(tmp_path, 0.06)
    assert first >= 30000, f"the aged plant differed at tick {first}, before its start"
    assert yellow == -1, f"L = 0.06 left yellow at tick {yellow}"
    g = 1.0 - math.exp(-(130000 - 1 - 30000) / 20000.0)
    expected = -4.5 * 0.06 * g * soc
    # SoC itself moves a little under the ageing, so the bus drop is checked to 25%.
    assert expected * 1.25 < dbus < expected * 0.75, (dbus, expected)


@needs_cxx
def test_at_zero_loss_the_aged_plant_is_the_balanced_plant(tmp_path: Path) -> None:
    first, yellow, dbus, _ = _run(tmp_path, 0.0)
    assert first == -1 and dbus == 0.0 and yellow == -1
