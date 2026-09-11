"""The ground toolkit, through the acceptance ladder (`docs/MODELS.md` 40).

Tier 1 is the generated fixture and tier 2 is the same path at one channel;
both run here. **Tier 3, a full mission, is not run from here** -- 40.13 makes
it a stop: no full fit is started until a single-channel wall clock is measured
and the options are costed. That rung is a gate, not a test.

The predictions these correspond to are 40.10's T1 to T8, and each test names
the one it adjudicates.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sentinel_toolkit import statistic  # noqa: E402
from sentinel_toolkit.calibrate import calibrate, derive_cut  # noqa: E402
from sentinel_toolkit.errors import (CalibrationError, ShapeError,  # noqa: E402
                                     TelemetryError, ToolkitError)
from sentinel_toolkit.fit import fit_model  # noqa: E402
from sentinel_toolkit.limits import (DEFAULT_EWMA_SPAN, FLIGHT_ERROR_WINDOW,  # noqa: E402
                                     FLIGHT_LIMITS)
from sentinel_toolkit.selftest import FIXTURE, healthy_run  # noqa: E402
from sentinel_toolkit.spec import to_spec  # noqa: E402
from sentinel_export import Status, read_model  # noqa: E402

ROUND_TRIP = ROOT / "flight" / "build" / "RoundTripDump"
needs_round_trip = pytest.mark.skipif(
    not ROUND_TRIP.exists(),
    reason="flight/build/RoundTripDump absent; run `make -C flight test`")


@pytest.fixture(scope="module")
def fixture_telemetry():
    values, names = healthy_run(seed=0, steps=120_000)
    return values, names


@pytest.fixture(scope="module")
def fitted(fixture_telemetry):
    values, names = fixture_telemetry
    return fit_model(values, channel_names=list(names), mission="fixture",
                     seed=0, **FIXTURE)


# -- the constants, checked rather than remembered ---------------------------

def test_the_flight_limits_match_config_hpp() -> None:
    """The dict in `src/` is a copy, so it is compared with its original.

    It existed only in `tests/test_model_file.py` before the toolkit, which is
    why the toolkit could have quietly become its third copy.
    """
    text = (ROOT / "flight" / "include" / "sentinel" / "Config.hpp").read_text()
    found = {name: int(re.search(rf"constexpr\s+U32\s+{name}\s*=\s*(\d+)U", text).group(1))
             for name in ("MAX_CHANNELS", "MAX_LAYERS", "MAX_HIDDEN", "MAX_PREDICTIONS")}
    found["MAX_INPUTS"] = found["MAX_CHANNELS"]          # Config.hpp:36 defines it so
    assert FLIGHT_LIMITS == found, (
        f"src/sentinel_toolkit/limits.py says {FLIGHT_LIMITS}; Config.hpp says {found}")


def test_the_trailing_span_is_the_one_the_component_compiles_in() -> None:
    """(!) It is not a model-file field, so the toolkit does not get to pick it.

    `DerivativeStream::configure` takes `Config::ERROR_WINDOW` directly and
    `Detector::configure` passes it nothing from the file, so a calibration at
    any other span derives a cut for a statistic the component does not compute.
    """
    config = (ROOT / "flight" / "include" / "sentinel" / "Config.hpp").read_text()
    batch = int(re.search(r"ERROR_WINDOW_BATCH\s*=\s*(\d+)U", config).group(1))
    count = int(re.search(r"ERROR_WINDOW_COUNT\s*=\s*(\d+)U", config).group(1))
    assert FLIGHT_ERROR_WINDOW == batch * count

    derivative = (ROOT / "flight" / "src" / "DerivativeStream.cpp").read_text()
    assert "Config::ERROR_WINDOW" in derivative, (
        "DerivativeStream no longer takes its span from Config; if it became a "
        "model-file field, the toolkit may choose it and this rule is stale")


def test_the_moved_statistic_is_the_arm_that_was_measured() -> None:
    """`trailing_stats` and `zstat` moved out of `scripts/decision_layer_arms.py`.

    Pinned against the definitions that file carried when D65 measured the arm,
    transcribed here, so the move is checkable rather than asserted.
    """
    def historical_trailing(x, span):
        x = np.asarray(x, dtype=np.float64); n = len(x)
        c1 = np.concatenate([[0.0], np.cumsum(x)])
        c2 = np.concatenate([[0.0], np.cumsum(x * x)])
        lo = np.maximum(np.arange(n) + 1 - span, 0); cnt = np.arange(n) + 1 - lo
        s1 = c1[np.arange(n) + 1] - c1[lo]; s2 = c2[np.arange(n) + 1] - c2[lo]
        mu = s1 / cnt
        return mu, np.sqrt(np.maximum(s2 / cnt - mu * mu, 0.0))

    x = np.random.default_rng(11).normal(size=6000) ** 2
    mu_a, sd_a = historical_trailing(x, FLIGHT_ERROR_WINDOW)
    mu_b, sd_b = statistic.trailing_stats(x, FLIGHT_ERROR_WINDOW)
    assert np.array_equal(mu_a, mu_b) and np.array_equal(sd_a, sd_b)

    z_historical = (x - mu_a) / np.maximum(sd_a, 1e-12)
    assert np.array_equal(z_historical, statistic.zstat(x, FLIGHT_ERROR_WINDOW))
    assert statistic.derivative(x)[0] == 0.0


# -- T2: the file is version 2 and round-trips ------------------------------

def test_t2_the_written_model_is_version_two_and_reads_back(fitted) -> None:
    status, spec = read_model(fitted.model_bytes, FLIGHT_LIMITS)
    assert status is Status.OK
    params = spec["params"]
    assert params["param_version"] == 2, (
        "a fused z-score cut in a version-1 file is the silent drift "
        "docs/MODEL_FILE.md 6.2 forbids (40.6)")
    assert params["tier"] == 3 and params["baseline_only"] is False
    assert params["ewma_span"] == DEFAULT_EWMA_SPAN
    assert params["threshold"] == fitted.calibration.cut
    assert params["warmup_steps"] == FIXTURE["window"] + FLIGHT_ERROR_WINDOW


def test_the_writer_refuses_to_guess_the_generation() -> None:
    """`to_spec` has no default for `param_version`, and that is the point."""
    with pytest.raises(TypeError):
        to_spec(object(), 1.0, 10, "x")                       # type: ignore[call-arg]


@needs_round_trip
def test_t1_the_model_round_trips_through_the_flight_loader(fitted, tmp_path) -> None:
    """The C++ leg: the flight loader re-emits the toolkit's own file exactly."""
    source = tmp_path / "toolkit.bin"
    source.write_bytes(fitted.model_bytes)
    out = tmp_path / "toolkit.rt"
    result = subprocess.run([str(ROUND_TRIP), str(source), str(out)],
                            cwd=str(ROOT / "flight"), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.read_bytes() == source.read_bytes(), (
        "the flight loader did not re-emit the toolkit's model byte-identically")


# -- T3: the refusals -------------------------------------------------------

def test_t3_every_refusal_fires_and_names_its_reason() -> None:
    good = np.zeros((5000, 3), dtype=np.float32)

    with pytest.raises(TelemetryError, match="dimensional"):
        fit_model(np.zeros(100, dtype=np.float32))
    with pytest.raises(ShapeError, match="MAX_CHANNELS|at most"):
        fit_model(np.zeros((5000, FLIGHT_LIMITS["MAX_CHANNELS"] + 1), dtype=np.float32))
    bad = good.copy(); bad[17, 1] = np.nan
    with pytest.raises(TelemetryError, match="non-finite"):
        fit_model(bad)
    with pytest.raises(TelemetryError, match="nothing to fit on"):
        fit_model(np.zeros((40, 2), dtype=np.float32), window=50, n_predictions=5)
    with pytest.raises(CalibrationError, match="trailing window"):
        fit_model(good, **FIXTURE)          # 5,000 steps cannot hold two 2,100 halves


def test_t3_no_refusal_fires_on_the_valid_fixture(fitted) -> None:
    """A toolkit that refuses good data is worse than one that accepts bad."""
    assert fitted.model_bytes and fitted.calibration.cut > 0.0


# -- T4: determinism --------------------------------------------------------

def test_t4_two_runs_produce_byte_identical_models(fixture_telemetry) -> None:
    values, names = fixture_telemetry
    a = fit_model(values, channel_names=list(names), mission="fixture", seed=0, **FIXTURE)
    b = fit_model(values, channel_names=list(names), mission="fixture", seed=0, **FIXTURE)
    assert a.model_bytes == b.model_bytes, (
        "a calibration that cannot be repeated cannot be cited")


# -- T5: the calibration is label-free, structurally --------------------------

def test_t5_the_toolkit_cannot_see_a_label() -> None:
    """Not "does not"; **cannot**. There is no parameter for one.

    The central claim of the report is that an operator with **no labelled
    anomalies** can read it. If a label reached the calibration, the claim would
    be false -- so the check is structural rather than behavioural, and it reads
    the parsed code rather than the prose: the docstrings talk about labels
    constantly, because saying what is not consulted is most of the point.

    `selftest.py` is excluded and named: it uses the fixture's own truth to pick
    the longest healthy run, which is how the fixture is *built*. Nothing it
    learns there reaches `fit_model`, which is handed an array and nothing else.
    """
    import ast
    import inspect

    banned = {"label", "labels", "truth", "anomaly", "anomalies", "LabelSet"}
    for name in ("calibrate", "fit", "statistic", "validate", "spec", "report"):
        tree = ast.parse((SRC / "sentinel_toolkit" / f"{name}.py").read_text())
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                seen.add(node.id)
            elif isinstance(node, ast.Attribute):
                seen.add(node.attr)
            elif isinstance(node, ast.arg):
                seen.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                seen.update(a.name for a in node.names)
                seen.add(getattr(node, "module", "") or "")
        leaked = seen & banned
        assert not leaked, f"sentinel_toolkit.{name} handles {sorted(leaked)}"

    parameters = set(inspect.signature(fit_model).parameters)
    assert not parameters & banned


# -- T6: the cut generalises on the mission's own data ------------------------

def test_t6_the_held_out_rate_is_within_twice_the_calibration_rate(fitted) -> None:
    """37.6's R2.5 band, on the same reasoning, and it is the only evidence
    available without labels that the chosen quantile transfers."""
    ratio = fitted.calibration.ratio
    assert 0.5 <= ratio <= 2.0, (
        f"held-out rate is {ratio:.2f}x the calibration half's; outside 2x means "
        "the quantile does not survive one split of one mission's nominal data")


def test_the_cut_is_derived_and_no_target_rate_is_accepted() -> None:
    """docs/HARNESS.md 1: the threshold is a noise floor, not a dial."""
    import inspect
    source = inspect.getsource(sys.modules["sentinel_toolkit.calibrate"])
    assert "target" not in source.lower() or "no target" in source.lower()
    assert "rate" not in inspect.signature(fit_model).parameters
    scores = np.random.default_rng(5).normal(size=20_000)
    assert derive_cut(scores, 0.999) > derive_cut(scores, 0.99)


def test_the_calibration_never_uses_one_half_twice() -> None:
    fit_half = np.zeros(3000)
    holdout = np.ones(3000)
    c = calibrate(fit_half, holdout, span=FLIGHT_ERROR_WINDOW)
    assert c.fit_steps == 3000 and c.holdout_steps == 3000
    assert c.holdout_alarms == 3000 and c.fit_alarms == 3000


# -- T7 and T8 ---------------------------------------------------------------

def test_t7_the_toolkit_reaches_no_bucket() -> None:
    """Zero cloud operations, and there is no client here to spend one."""
    package = SRC / "sentinel_toolkit"
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for banned in ("boto3", "sentinel_data", "r2", "botocore"):
            assert not re.search(rf"^\s*(import|from)\s+{banned}\b", text, re.M), (
                f"{path.name} imports {banned}")


def test_t8_provenance_is_written_and_distinguishes_two_calibrations() -> None:
    a = calibrate(np.random.default_rng(1).normal(size=4000),
                  np.random.default_rng(2).normal(size=4000), span=FLIGHT_ERROR_WINDOW)
    b = calibrate(np.random.default_rng(1).normal(size=9000),
                  np.random.default_rng(2).normal(size=9000), span=FLIGHT_ERROR_WINDOW)
    assert a.provenance() and b.provenance()
    assert a.provenance() != b.provenance(), (
        "D29: a threshold is a parameter with a provenance, and two calibrations "
        "on different windows must not record the same one")
    assert len(b.provenance().encode("ascii")) < 64


# -- the ladder --------------------------------------------------------------

def test_tier_two_of_the_ladder_is_one_channel(fixture_telemetry) -> None:
    """The univariate shape 40.3 names: the multivariate one at n_channels 1."""
    values, names = fixture_telemetry
    result = fit_model(values[:, :1], channel_names=[names[0]], mission="fixture",
                       seed=0, **FIXTURE)
    status, spec = read_model(result.model_bytes, FLIGHT_LIMITS)
    assert status is Status.OK and spec["n_channels"] == 1
    assert spec["params"]["param_version"] == 2


def test_the_one_command_runs_end_to_end(tmp_path, fixture_telemetry) -> None:
    """`python -m sentinel_toolkit fit`, which is what "one command" means here."""
    values, _names = fixture_telemetry
    telemetry = tmp_path / "healthy.npy"
    np.save(telemetry, values[:, :2])
    out = tmp_path / "model.bin"
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-m", "sentinel_toolkit", "fit",
         "--telemetry", str(telemetry), "--out", str(out),
         "--window", "50", "--hidden", "16", "16",
         "--predictions", "5", "--epochs", "2"],
        cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert out.exists()
    assert "PRE-LAUNCH CALIBRATION REPORT" in result.stdout
    assert "REPORTED, NEVER TARGETED" in result.stdout
    status, spec = read_model(out.read_bytes(), FLIGHT_LIMITS)
    assert status is Status.OK and spec["params"]["param_version"] == 2


def test_a_refusal_exits_two_and_says_why(tmp_path) -> None:
    telemetry = tmp_path / "short.npy"
    np.save(telemetry, np.zeros((40, 2), dtype=np.float32))
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-m", "sentinel_toolkit", "fit",
         "--telemetry", str(telemetry), "--out", str(tmp_path / "m.bin"),
         "--window", "50", "--predictions", "5", "--epochs", "1"],
        cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "REFUSED:" in result.stderr
    assert not (tmp_path / "m.bin").exists(), "a refused run wrote a model anyway"
