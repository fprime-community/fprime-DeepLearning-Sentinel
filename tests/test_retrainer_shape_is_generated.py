"""D84 / `docs/MODELS.md` 77: the retrainer at a mission's shape is GENERATED, and re-earns its gates.

D83.1 route 1 is a second compile-time instantiation of `deep_f32.ml`, made by
substitution so *"the two cannot drift"*. That phrase is a claim, and these are
the checks that make it one rather than a hope:

1. **At the maxima the generator IS the template** -- byte for byte (SX1). If
   the substitution touched anything it should not, this is where it shows.
2. **At 8/10 exactly three lines move** -- `ins`, `n_out`, `n_params` -- and
   `n_params` is `src/sentinel_export/format.py`'s own count (SX2).
3. **The weight count identifies the shape** over all 160 admissible shapes
   (SX3). That is what makes `shadow59.ml`'s size refusal sufficient; without
   it the component would need its own header check, and D84 says which.
4. **Every generated file keeps the checked accessor and the template's
   `[@zero_alloc strict]` count** -- the static half of SX4 and of D82.
5. **The flying-file builder's default output is unchanged** by the
   `--channels`/`--predictions` it gained, pinned by hash.

These run on any tree. With the OxCaml switch present, `scripts/oxcaml_shape.sh`
runs every non-EX1 gate at both shapes (SX4 to SX7) into a temporary directory;
without it this SKIPS loudly (D79). EX1's committed logs are parsed separately.

Every check here has its negative direction beside it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "oxcaml" / "retrainer" / "deep_f32.ml"
SHAPE_SCRIPT = ROOT / "scripts" / "oxcaml_shape.sh"
SWITCH_OCAMLOPT = ROOT / "oxcaml" / ".opam" / "5.2.0+ox" / "bin" / "ocamlopt"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

sys.path.insert(0, str(ROOT / "src"))
from sentinel_export import format as fmt  # noqa: E402


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shape = _load("oxcaml_shape", ROOT / "scripts" / "oxcaml_shape.py")

#: What `scripts/s72_flying_file.py` wrote before it gained --channels and
#: --predictions, measured 2026-09-25 at e039054: 302,048 B.
S72_DEFAULT_SHA256 = "afff41836ee7127703297f2d22a901ed398ee68b3473fb380ae2d57413e761a5"

STRICT = re.compile(r"\[@+zero_alloc strict\]")
ASSUME = re.compile(r"\[@+zero_alloc[^]]*assume")


def _template() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


# -- SX1, SX2 ------------------------------------------------------------------------

def test_at_the_maxima_the_generator_is_the_template() -> None:
    assert shape.generate(16, 10, _template()) == _template(), (
        "scripts/oxcaml_shape.py at 16 channels, 10 predictions no longer reproduces "
        "oxcaml/retrainer/deep_f32.ml byte for byte. D83.1 route 1's 'the two cannot "
        "drift' rests on this.")


def test_the_identity_check_can_fail() -> None:
    """The negative direction: a generator that rewrites a line at the maxima is caught."""
    original = shape._line
    try:
        shape._line = lambda name, value, maxima, width: (
            f"let {name} = {value}".ljust(width) + "(* rewritten *)\n")
        assert shape.generate(16, 10, _template()) != _template()
    finally:
        shape._line = original


def test_at_the_mission_shape_exactly_three_lines_move() -> None:
    before = _template().splitlines()
    after = shape.generate(8, 10, _template()).splitlines()
    assert len(before) == len(after)
    moved = [(a, b) for a, b in zip(before, after) if a != b]
    names = [re.match(r"let (\w+) =", b).group(1) for _, b in moved]
    assert names == ["ins", "n_out", "n_params"], moved
    values = {re.match(r"let (\w+) = (\d+)", b).group(1): int(re.match(
        r"let (\w+) = (\d+)", b).group(2)) for _, b in moved}
    assert values == {"ins": 8, "n_out": 80, "n_params": 66960}


def test_an_anchor_that_does_not_match_exactly_once_is_refused() -> None:
    doubled = _template() + shape.ANCHORS["ins"]
    with pytest.raises(shape.ShapeError, match="matched 2 times"):
        shape.generate(8, 10, doubled)
    moved = _template().replace(shape.ANCHORS["n_out"], "let n_out = 160\n")
    with pytest.raises(shape.ShapeError, match="matched 0 times"):
        shape.generate(8, 10, moved)


# -- SX2, SX3: the count --------------------------------------------------------------

ADMISSIBLE = [(c, p) for c in range(1, 17) for p in range(1, 11)]


def test_the_count_is_the_format_packages_count_at_every_shape() -> None:
    for c, p in ADMISSIBLE:
        assert shape.parameter_count(c, p) == fmt.parameter_count(c, [80, 80], p, c), (c, p)
    assert shape.parameter_count(16, 10) == 75360
    assert shape.parameter_count(12, 10) == 71160
    assert shape.parameter_count(8, 10) == 66960


def test_the_count_is_deep_f32s_own_offsets() -> None:
    """`o_hb + n_out` in deep_f32.ml's offset arithmetic, at every shape."""
    hs, gw = 80, 240
    for c, p in ADMISSIBLE:
        n_out = p * c
        o_hb = (gw * c) + (3 * gw * hs) + (4 * gw) + (n_out * hs)
        assert o_hb + n_out == shape.parameter_count(c, p), (c, p)


def test_the_weight_count_identifies_the_shape() -> None:
    """SX3. Why `shadow59.ml:86`'s size refusal is enough, and the check that it is."""
    counts = [shape.parameter_count(c, p) for c, p in ADMISSIBLE]
    assert len(set(counts)) == len(counts) == 160, (
        "two admissible shapes share a weight count, so a flying file at one would pass "
        "shadow59.ml's size check with weights from the other. D84 says the component "
        "then needs its own header check.")


def test_the_uniqueness_check_can_fail() -> None:
    """Widen the box to where a collision exists, and the same check finds it."""
    wide = [(c, p) for c in range(1, 65) for p in range(1, 65)]
    counts = [shape.parameter_count(c, p) for c, p in wide]
    assert len(set(counts)) < len(counts)


@pytest.mark.parametrize("channels,predictions", [(0, 10), (17, 10), (8, 0), (8, 11)])
def test_a_shape_outside_the_box_is_refused(channels: int, predictions: int) -> None:
    with pytest.raises(shape.ShapeError, match="Config.hpp"):
        shape.generate(channels, predictions, _template())


# -- SX4 (static), D82 ----------------------------------------------------------------

@pytest.mark.parametrize("channels,predictions", [(8, 10), (16, 10), (1, 1), (12, 7)])
def test_every_generated_file_keeps_the_accessor_and_the_annotations(
        channels: int, predictions: int) -> None:
    text = shape.generate(channels, predictions, _template())
    assert re.search(r"^open Acc$", text, re.M), "the checked accessor was lost (D82)"
    assert len(STRICT.findall(text)) == len(STRICT.findall(_template())) > 0
    assert not ASSUME.findall(text)


def test_the_annotation_check_can_fail() -> None:
    stripped = _template().replace("[@zero_alloc strict]", "", 1)
    assert len(STRICT.findall(stripped)) != len(STRICT.findall(_template()))


def test_the_header_carries_the_same_numbers() -> None:
    h = shape.header(8, 10)
    assert "#define SENTINEL_CYCLE_CHANNELS 8U" in h
    assert "#define SENTINEL_CYCLE_PREDICTIONS 10U" in h
    assert "#define SENTINEL_CYCLE_WINDOW 250U" in h
    assert "#define SENTINEL_CYCLE_N_PARAMS 66960U" in h


# -- the flying file ------------------------------------------------------------------

def _flying(tmp_path: pathlib.Path, *args: str) -> bytes:
    pytest.importorskip("numpy")
    out = tmp_path / "fly.bin"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "s72_flying_file.py"),
                             str(out), *args], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return out.read_bytes()


def test_the_flying_files_default_output_is_unchanged(tmp_path: pathlib.Path) -> None:
    data = _flying(tmp_path)
    assert hashlib.sha256(data).hexdigest() == S72_DEFAULT_SHA256, (
        "scripts/s72_flying_file.py's default output moved. It is 72's HO1 apparatus; "
        "the --channels/--predictions it gained for D84 were to leave it untouched.")


def test_the_flying_file_at_the_mission_shape_is_the_flown_files_size(
        tmp_path: pathlib.Path) -> None:
    """268,224 B is `SentinelModel.bin`'s size (`docs/MODELS.md` 72.10)."""
    data = _flying(tmp_path, "--channels", "8", "--predictions", "10")
    assert len(data) == 268224
    assert int.from_bytes(data[36:40], "little") // 4 == 66960


# -- SX4 to SX7, live -----------------------------------------------------------------

needs_switch = pytest.mark.skipif(
    not SWITCH_OCAMLOPT.exists() or not VENV_PYTHON.exists(),
    reason="no OxCaml switch or no .venv; run scripts/oxcaml_setup.sh")


@needs_switch
@pytest.mark.parametrize("channels,predictions", [(8, 10), (16, 10)])
def test_every_gate_holds_at_the_generated_shape(
        tmp_path: pathlib.Path, channels: int, predictions: int) -> None:
    out = tmp_path / f"shape-c{channels}-p{predictions}"
    env = dict(os.environ, OXCAML_SHAPE_OUT=str(out))
    env.pop("EX1", None)
    result = subprocess.run(["bash", str(SHAPE_SCRIPT), "--channels", str(channels),
                             "--predictions", str(predictions)],
                            capture_output=True, text=True, env=env, timeout=3600)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-4000:]
    for gate in ("strict HOLDS", "rejected:", "RAISED: Invalid_argument",
                 "HO1 at this shape: all checks passed",
                 f"== shape c{channels}-p{predictions}: every gate passed =="):
        assert gate in log, (gate, log[-4000:])


# -- SX8, SX9: EX1, from the committed run logs -----------------------------------------
#
# EX1 is sharded across ten processes and is not re-run by the suite; the full gate
# log of each shape's run is committed, as `tests/fixtures/oxcaml_e3_x8_rejection.log`
# is for E3's X8, and these checks make the log evidence rather than decoration.

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _ex1(name: str) -> dict:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    m = re.search(r"EXHAUSTIVE: checked ([\d,]+)\s+outside (\d+)\s+worst ratio ([\d.]+) at idx "
                  r"(\d+)\s+worst \|err\| ([\d.e+-]+)", text)
    assert m, f"{name} carries no EXHAUSTIVE line"
    return {"text": text, "checked": int(m.group(1).replace(",", "")),
            "outside": int(m.group(2)), "ratio": m.group(3), "idx": int(m.group(4)),
            "worst": m.group(5)}


def _ex1_faults(run: dict, n_params: int) -> list[str]:
    faults = []
    if run["checked"] != n_params:
        faults.append(f"checked {run['checked']}, not every one of {n_params}")
    if run["outside"] != 0:
        faults.append(f"{run['outside']} outside the band")
    if "shard 0 of 1021 reproduces deep_f32_check.ml at this shape: True" not in run["text"]:
        faults.append("the shard-0 validation did not hold")
    if "EX1 -> HOLD" not in run["text"]:
        faults.append("no HOLD verdict")
    for gate in ("strict HOLDS", "rejected:", "RAISED: Invalid_argument",
                 "HO1 at this shape: all checks passed"):
        if gate not in run["text"]:
            faults.append(f"the log lacks '{gate}'")
    return faults


def test_ex1_holds_at_the_mission_shape() -> None:
    """SX8: 66,960 = `SentinelRef`'s flown shape (8 channels, 10 predictions)."""
    run = _ex1("oxcaml_shape_c8_p10_gates.log")
    assert not _ex1_faults(run, shape.parameter_count(8, 10)), _ex1_faults(run, 66960)


def test_ex1_at_the_maxima_reproduces_section_64() -> None:
    """SX9: the generated maxima are the template, so 64's figures must come back exactly."""
    run = _ex1("oxcaml_shape_c16_p10_gates.log")
    assert not _ex1_faults(run, 75360), _ex1_faults(run, 75360)
    assert (run["ratio"], run["idx"], run["worst"]) == ("0.306952", 11365, "8.819351553e-05"), (
        "EX1 at the generated maxima no longer reproduces docs/MODELS.md 64 "
        "(worst err/allowed 0.306952 at index 11365)")


def test_the_ex1_log_check_can_fail() -> None:
    run = _ex1("oxcaml_shape_c8_p10_gates.log")
    assert _ex1_faults(dict(run, outside=1), 66960)
    assert _ex1_faults(dict(run, checked=66959), 66960)
    assert _ex1_faults(dict(run, text=run["text"].replace("EX1 -> HOLD", "EX1 -> FAIL")), 66960)
