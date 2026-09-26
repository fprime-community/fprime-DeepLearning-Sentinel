"""D85 / `docs/MODELS.md` 78: the retrainer LEARNS, the plant it learns from is what 78.3
says, and the loop topology is generated from the one topology there is.

Until D85 the flown cycle optimised 57's head-only TEST functional against no target,
reset Adam on every call, and never warm-started; the testbed's default plant drained
its battery on a healthy run. Each is now checked, in both directions:

1. **The flown loss has a target.** `deep_f32.ml` carries `tgt` and the MSE terms;
   `cycle_c.ml`'s window is (t_max + P) rows; `run_cycle` no longer resets Adam; the
   warm start and the seed do. Static, always runs.
2. **The testbed's DEFAULT plant is bit-identical** to what 42.9 measured on, pinned by
   a hash of its channel values, healthy and faulted, seeds 1 and 7. And 78.3's
   scenarios hold: the balanced plant and the aged plant stay inside yellow, while 10%
   solar loss drains the battery (the negative direction). Needs a C++ compiler.
3. **SentinelRef's loop topology is generated from `topology.fpp`** by two anchors that
   must each appear exactly once. Static.
4. **EX1 at 8/10 under the MSE loss HELD**, read from the committed D85 log.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RETRAINER = ROOT / "oxcaml" / "retrainer"
POWERSIM = ROOT / "fprime" / "SentinelRef" / "PowerSim"
TOP = ROOT / "fprime" / "SentinelRef" / "Top"
FIXTURES = Path(__file__).parent / "fixtures"

#: 78.3: the channel values of the DEFAULT plant, healthy and faulted, seeds 1 and 7,
#: 30,000 ticks, FNV-1a over the raw doubles. Measured before and after D85's change.
DEFAULT_PLANT_HASH = "9d72497a2ff277b3"


# -- 1. the flown loss has a target ------------------------------------------------------

def test_the_flown_cycle_trains_against_a_target() -> None:
    deep = (RETRAINER / "deep_f32.ml").read_text()
    assert re.search(r"^let tgt = mk n_out$", deep, re.M), "deep_f32.ml has no target buffer"
    assert "sub !acc (get tgt i)" in deep, "the forward pass does not compare to the target"
    assert "sub (get y i) (get tgt i)" in deep, "the backward pass does not use (y - target)"
    cyc = (RETRAINER / "cycle_c.ml").read_text()
    assert "n <> n_in + Deep_f32.n_out" in cyc, "cycle_c.ml's window does not carry the target"


def test_adam_is_reset_where_a_run_begins_and_not_per_call() -> None:
    deep = (RETRAINER / "deep_f32.ml").read_text()
    body = deep[deep.index("let[@zero_alloc strict] run_cycle"):]
    assert "adam_reset ()" not in body.split("\n\n")[0], (
        "run_cycle resets Adam on every call again: at one step per tick that is sign "
        "descent (D85)")
    assert "Deep_f32.adam_reset ()" in (RETRAINER / "shadow_c.ml").read_text()
    assert "Deep_f32.adam_reset ()" in (RETRAINER / "cycle_c.ml").read_text()


def test_the_target_check_can_fail() -> None:
    mutated = (RETRAINER / "deep_f32.ml").read_text().replace("let tgt = mk n_out", "")
    assert not re.search(r"^let tgt = mk n_out$", mutated, re.M)


# -- 2. the plant ------------------------------------------------------------------------

needs_cxx = pytest.mark.skipif(shutil.which("clang++") is None and shutil.which("c++") is None,
                               reason="no C++ compiler")

HASH_SRC = r'''
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include "PowerPlant.hpp"
using namespace Testbed;
int main() {
    unsigned long long h = 1469598103934665603ULL;
    for (uint32_t seed : {1u, 7u}) for (int f = 0; f < 2; ++f) {
        PowerPlant p; p.reset(seed);
        for (uint32_t t = 0; t < 30000; ++t) {
            p.step(f && t >= 8000, 2.0e-4);
            for (uint32_t c = 0; c < PLANT_CHANNELS; ++c) {
                double v = p.value(c); unsigned char b[8]; std::memcpy(b, &v, 8);
                for (int i = 0; i < 8; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
            }
        }
    }
    std::printf("%016llx\n", h);
}
'''

ENVELOPE_SRC = r'''
#include <cstdio>
#include <cstdlib>
#include "PowerPlant.hpp"
using namespace Testbed;
static const double YLO[8] = {-2.0,-10.0,0.5,27.0,-5.0,-35.0,0.0,0.30};
static const double YHI[8] = {300.0,10.0,9.5,32.0,40.0,30.0,1.0,1.0};
int main(int argc, char** argv) {
    const double solar = std::atof(argv[1]), emis = std::atof(argv[2]);
    PowerPlant p; p.reset(1u); p.configureBalance(0.60, 0.92);
    if (solar > 0.0 || emis > 0.0) p.configureAgeing(60000u + 5000u, 8000.0, solar, emis);
    double socLate = 0.0, socEnd = 0.0; long yellow = -1;
    // SoC is compared at the SAME orbital phase, ten orbits apart (5,400 ticks each):
    // an instantaneous reading swings by about 0.04 within one orbit.
    for (uint32_t t = 0; t <= 60000u + 108000u; ++t) {
        p.step(false, 0.0);
        if (t < 60000u) continue;
        for (uint32_t c = 0; c < PLANT_CHANNELS; ++c) {
            const double v = p.value(c);
            if (yellow < 0 && (v < YLO[c] || v > YHI[c])) yellow = (long)t;
        }
        if (t == 60000u + 54000u) socLate = p.value(CH_STATE_OF_CHARGE);
        if (t == 60000u + 108000u) socEnd = p.value(CH_STATE_OF_CHARGE);
    }
    std::printf("%ld %.6f\n", yellow, socEnd - socLate);
}
'''


def _build(tmp_path: Path, name: str, src: str) -> Path:
    cxx = shutil.which("clang++") or shutil.which("c++")
    (tmp_path / f"{name}.cpp").write_text(src)
    exe = tmp_path / name
    r = subprocess.run([cxx, "-std=c++14", "-O2", f"-I{POWERSIM}", str(tmp_path / f"{name}.cpp"),
                        str(POWERSIM / "PowerPlant.cpp"), "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


@needs_cxx
def test_the_default_plant_is_what_42_measured_on(tmp_path: Path) -> None:
    out = subprocess.run([str(_build(tmp_path, "h", HASH_SRC))], capture_output=True, text=True)
    assert out.stdout.strip() == DEFAULT_PLANT_HASH, (
        "the default plant's trace moved. 42.9's figures were measured on it; D85's modes "
        "are opt-in precisely so it would not.")


@needs_cxx
def test_the_balanced_and_aged_plants_stay_healthy_and_the_drain_is_seen(tmp_path: Path) -> None:
    exe = _build(tmp_path, "e", ENVELOPE_SRC)
    def run(solar: float, emis: float) -> tuple[int, float]:
        y, drift = subprocess.run([str(exe), str(solar), str(emis)], capture_output=True,
                                  text=True).stdout.split()
        return int(y), float(drift)
    for solar, emis in ((0.0, 0.0), (0.05, 0.30)):
        yellow, drift = run(solar, emis)
        assert yellow == -1, f"solar {solar}, emis {emis}: left yellow at tick {yellow}"
        assert abs(drift) < 0.01, f"solar {solar}: SoC drifted {drift} over ten orbits"
    # The negative direction, 78.3's own finding: 10% solar loss drains the battery.
    _, drift = run(0.10, 0.30)
    assert drift < -0.01, f"10% solar loss should drain SoC; drift was {drift}"


# -- 3. the loop topology's anchors ------------------------------------------------------

ANCHORS = ("    instance powerSim\n", "      powerSim.channelOut -> sentinelMonitor.channelsIn\n")


def _anchor_counts(topology: str) -> list[int]:
    return [topology.count(a) for a in ANCHORS]


def test_the_loop_topology_can_be_generated_from_the_one_topology() -> None:
    assert _anchor_counts((TOP / "topology.fpp").read_text()) == [1, 1]
    cmake = (TOP / "CMakeLists.txt").read_text()
    for a in ANCHORS:
        assert a.rstrip("\n") in cmake, f"Top/CMakeLists.txt no longer substitutes {a!r}"


def test_the_anchor_check_can_fail() -> None:
    mutated = (TOP / "topology.fpp").read_text().replace(ANCHORS[1], "")
    assert _anchor_counts(mutated) == [1, 0]


# -- 4. EX1 under the MSE loss -----------------------------------------------------------

@pytest.mark.parametrize("name,n_params", [("oxcaml_shape_c8_p10_d85_gates.log", 66960),
                                           ("oxcaml_shape_c16_p10_d85_gates.log", 75360),
                                           ("oxcaml_shape_c12_p10_d85_gates.log", 71160)])
def test_ex1_holds_under_the_flown_loss(name: str, n_params: int) -> None:
    text = (FIXTURES / name).read_text()
    m = re.search(r"EXHAUSTIVE: checked ([\d,]+)\s+outside (\d+)", text)
    assert m, f"{name} carries no EXHAUSTIVE line"
    assert int(m.group(1).replace(",", "")) == n_params and int(m.group(2)) == 0
    for gate in ("EX1 -> HOLD", "shard 0 of 1021 reproduces deep_f32_check.ml at this shape: True",
                 "window56: strict HOLDS", "window_c: strict HOLDS",
                 "loss at step 1",
                 "the cycle at this shape: all checks passed"):
        assert gate in text, (name, gate)
