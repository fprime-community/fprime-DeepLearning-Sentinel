"""D84 / `docs/MODELS.md` 77, SX14: the loop, closed on the host, both ways.

72 wrote a candidate `flight/`'s reader loads, and 73.8 commanded it into
`SentinelRef` -- where it was REFUSED, correctly, because the cycle was 16 channels
and the deployment 8. D84 generates the cycle at the mission's shape, and this is
the check that the two now meet:

1. `Sentinel_Retrainer_ut_exe`, built at the shape the build cache names, boots
   the OCaml runtime, runs the cycle and writes `loop_candidate.bin` from the
   cycle's OWN weights into a flying file at that shape.
2. `Sentinel_Monitor_ut_exe`, configured on that flying file, receives
   `RELOAD_MODEL loop_candidate.bin`: **accepted**, and the Monitor then runs on it
   -- twenty ticks, every one scored in `Mode::MODEL`, nothing degraded.
3. The other direction, twice: a valid file at another width is **refused by
   width**, and the retrainer's candidate with one byte flipped is **refused by
   CRC** -- each rolled back to the model that was flying.

And the check that the accept path itself can fail: the flipped candidate offered
as an `accept` must FAIL the Monitor's test.

(!) WHAT THIS IS NOT, NAMED BEFORE ANY READER ASKS. It proves the shapes, the file
format and the command meet. The candidate was trained for one step on the
retrainer's deterministic drive, which is NOT telemetry, so nothing about its
quality follows; nothing scored it before the command; and no shadow model may be
swapped in operationally (D83 c.4, D84 c.2). Skips loudly without the build trees.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UT_CACHE = ROOT / "fprime" / "build-fprime-automatic-native-ut"
UT_BIN = UT_CACHE / "bin" / "Darwin"
RETRAINER_UT = UT_BIN / "Sentinel_Retrainer_ut_exe"
MONITOR_UT = UT_BIN / "Sentinel_Monitor_ut_exe"


def _cache_value(name: str) -> str | None:
    cache = UT_CACHE / "CMakeCache.txt"
    if not cache.exists():
        return None
    m = re.search(rf"^{name}:\w+=(\S+)$", cache.read_text(encoding="utf-8", errors="replace"),
                  re.M)
    return m.group(1) if m else None


def _shape() -> tuple[int, int]:
    if (_cache_value("SENTINEL_WITH_RETRAINER") or "").upper() not in ("ON", "TRUE", "1"):
        pytest.skip("the unit-test cache is not configured with SENTINEL_WITH_RETRAINER=ON")
    for exe in (RETRAINER_UT, MONITOR_UT):
        if not exe.exists():
            pytest.skip(f"{exe.name} is not built; run fprime-util check in its directory")
    pytest.importorskip("numpy")
    return (int(_cache_value("SENTINEL_RETRAINER_CHANNELS")),
            int(_cache_value("SENTINEL_RETRAINER_PREDICTIONS")))


def _flying(out: pathlib.Path, channels: int, predictions: int) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s72_flying_file.py"), str(out),
                        "--channels", str(channels), "--predictions", str(predictions)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr


def _monitor(loop: pathlib.Path, candidate: str, channels: int, expect: str):
    env = dict(os.environ, SENTINEL_LOOP_DIR=str(loop), SENTINEL_LOOP_CANDIDATE=candidate,
               SENTINEL_LOOP_CHANNELS=str(channels), SENTINEL_LOOP_EXPECT=expect)
    r = subprocess.run([str(MONITOR_UT), "--gtest_filter=Loop.*"], capture_output=True,
                       text=True, env=env, cwd=str(loop), timeout=600)
    return r.returncode, r.stdout + r.stderr


@pytest.fixture(scope="module")
def loop(tmp_path_factory):
    channels, predictions = _shape()
    d = tmp_path_factory.mktemp("loop")
    _flying(d / "loop_flying.bin", channels, predictions)
    env = dict(os.environ, SENTINEL_LOOP_DIR=str(d))
    r = subprocess.run([str(RETRAINER_UT)], capture_output=True, text=True, env=env,
                       cwd=str(d), timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    cand = d / "loop_candidate.bin"
    assert cand.exists(), "the retrainer wrote no candidate"
    assert cand.stat().st_size == (d / "loop_flying.bin").stat().st_size
    # The two refusals' inputs.
    other = 16 if channels != 16 else 8
    _flying(d / "loop_wide.bin", other, predictions)
    data = bytearray(cand.read_bytes())
    data[64 + (20 * channels)] ^= 0xFF      # first weight byte, under static_crc32
    (d / "loop_flipped.bin").write_bytes(bytes(data))
    return d, channels


def test_the_retrainers_candidate_is_accepted_and_flown(loop) -> None:
    d, channels = loop
    rc, log = _monitor(d, "loop_candidate.bin", channels, "accept")
    assert rc == 0, log[-4000:]
    assert "[  PASSED  ] 1 test" in log and "SKIPPED" not in log, log[-2000:]


def test_a_candidate_at_another_width_is_refused(loop) -> None:
    d, channels = loop
    rc, log = _monitor(d, "loop_wide.bin", channels, "width")
    assert rc == 0 and "[  PASSED  ] 1 test" in log, log[-4000:]


def test_a_corrupted_candidate_is_refused_and_rolled_back(loop) -> None:
    d, channels = loop
    rc, log = _monitor(d, "loop_flipped.bin", channels, "crc")
    assert rc == 0 and "[  PASSED  ] 1 test" in log, log[-4000:]


def test_the_accept_path_can_fail(loop) -> None:
    """The negative direction of the first test: a candidate that must not load, offered
    as one that should, fails the Monitor's own assertions."""
    d, channels = loop
    rc, log = _monitor(d, "loop_flipped.bin", channels, "accept")
    assert rc != 0 and "[  FAILED  ]" in log, log[-4000:]
