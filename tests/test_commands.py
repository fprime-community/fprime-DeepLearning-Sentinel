"""Telecommands on the grid: impulses, not levels, and the trace derived from them.

telemanom feeds its LSTM telemetry **and encoded command information**
(`docs/DECISIONS.md` D6). Work item 4 reproduced the method without it, which is
why the false-alarm rate is what it is. These tests cover the half that was
missing.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval import commands
from sentinel_eval.errors import TaskError
from sentinel_eval.grid import Grid
from sentinel_eval.read import Executions
from sentinel_models.windows import command_features, decay

EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")
PERIOD = np.timedelta64(30, "s")


def _grid(n=100) -> Grid:
    return Grid(start=EPOCH, period=PERIOD, n=n)


def _executions(pairs) -> Executions:
    ids = np.array([name for name, _ in pairs])
    times = np.array([EPOCH + PERIOD * step for _, step in pairs], dtype="datetime64[ns]")
    return Executions(ids, times, np.ones(len(pairs), dtype=np.float32))


# -- rasterisation ----------------------------------------------------------
def test_a_command_occupies_one_cell_and_is_not_held_forward():
    """The distinction from every other series the harness puts on the grid.

    Telemetry is a level and `resample_zoh` holds it until the next sample. A
    command is an instant, and holding it forward would assert the spacecraft is
    being commanded continuously from the first execution to the end of mission.
    """
    out = commands.rasterise(_executions([("tc_a", 10)]), _grid(), ["tc_a"])
    assert out.shape == (100, 1)
    assert out[10, 0] == 1
    assert out[:10, 0].sum() == 0 and out[11:, 0].sum() == 0


def test_a_cell_carries_a_count_not_a_flag():
    """Two executions in one cell is a different event from one."""
    out = commands.rasterise(_executions([("tc_a", 5), ("tc_a", 5), ("tc_a", 5)]),
                             _grid(), ["tc_a"])
    assert out[5, 0] == 3


def test_columns_follow_the_order_given_not_the_data():
    """The model's input width is fixed at fit time; column order must be too."""
    executions = _executions([("tc_b", 3), ("tc_a", 7)])
    out = commands.rasterise(executions, _grid(), ["tc_a", "tc_b"])
    assert out[7, 0] == 1 and out[3, 1] == 1
    assert out[3, 0] == 0 and out[7, 1] == 0


def test_an_execution_outside_the_grid_is_dropped_not_clamped():
    """Clamping would pile every pre-grid command onto timestep zero.

    That invents a burst which never happened, at exactly the point where a model
    has no history to judge it against.
    """
    before = Executions(np.array(["tc_a"]),
                        np.array([EPOCH - PERIOD * 50], dtype="datetime64[ns]"),
                        np.ones(1, dtype=np.float32))
    after = Executions(np.array(["tc_a"]),
                       np.array([EPOCH + PERIOD * 500], dtype="datetime64[ns]"),
                       np.ones(1, dtype=np.float32))
    for executions in (before, after):
        assert commands.rasterise(executions, _grid(), ["tc_a"]).sum() == 0


def test_a_command_with_no_executions_is_an_all_zero_column_not_an_error():
    out = commands.rasterise(_executions([("tc_a", 4)]), _grid(), ["tc_a", "tc_dead"])
    assert out[:, 1].sum() == 0 and out[4, 0] == 1


def test_rasterising_nothing_is_refused():
    with pytest.raises(TaskError, match="no telecommands selected"):
        commands.rasterise(_executions([("tc_a", 1)]), _grid(), [])


# -- selection --------------------------------------------------------------
DESCRIPTIONS = [
    {"mission": "m1", "Telecommand": "tc_a", "Priority": 3},
    {"mission": "m1", "Telecommand": "tc_b", "Priority": 3},
    {"mission": "m1", "Telecommand": "tc_dead", "Priority": 3},
    {"mission": "m1", "Telecommand": "tc_low", "Priority": 1},
    {"mission": "m2", "Telecommand": "tc_a", "Priority": 3},
]


def test_selection_takes_the_graded_commands_of_one_mission():
    """The key is (mission, Telecommand): both missions reuse the same names."""
    chosen = commands.select(DESCRIPTIONS, "m1", 3, {"tc_a", "tc_b", "tc_dead", "tc_low"})
    assert chosen == ["tc_a", "tc_b", "tc_dead"]


def test_a_command_that_never_executes_is_dropped():
    """A constant-zero input column dilutes the features and weakens the ablation.

    17 of Mission1's 698 telecommands are declared and never fire.
    """
    chosen = commands.select(DESCRIPTIONS, "m1", 3, {"tc_a", "tc_b"})
    assert chosen == ["tc_a", "tc_b"]


def test_an_absent_priority_grade_is_refused_rather_than_returning_nothing():
    with pytest.raises(TaskError, match="no priority-2 telecommands"):
        commands.select(DESCRIPTIONS, "m1", 2, {"tc_a"})


# -- the derived trace ------------------------------------------------------
def test_the_decay_matches_the_closed_form():
    impulses = np.zeros((200, 1), dtype=np.uint8)
    impulses[50, 0] = 1
    traced = decay(impulses, steps=60)
    for after in (0, 20, 60, 120):
        assert traced[50 + after, 0] == pytest.approx(np.exp(-after / 60), rel=1e-5)


def test_overlapping_commands_take_the_maximum_not_the_sum():
    """The feature answers *how recently*; the impulse column answers *how many*.

    Summing would build a level the model has never seen during training and
    would then meet in flight the first time two commands landed close together.
    """
    impulses = np.zeros((100, 1), dtype=np.uint8)
    impulses[10, 0] = 1
    impulses[12, 0] = 1
    assert decay(impulses, steps=60).max() == pytest.approx(1.0)


def test_features_are_the_impulse_and_the_trace_side_by_side():
    impulses = np.zeros((500, 2), dtype=np.uint8)
    impulses[100, 0] = 1
    out = command_features(impulses, np.array([90]), window=40, decay_steps=60)
    assert out.shape == (1, 40, 4)                 # 2 commands x (impulse, trace)
    assert out[0, 10, 0] == 1.0                    # impulse, at absolute step 100
    assert out[0, 10, 2] == pytest.approx(1.0)     # trace, freshly fired
    assert out[0, 30, 0] == 0.0                    # no second impulse
    assert out[0, 30, 2] == pytest.approx(np.exp(-20 / 60), rel=1e-5)


def test_a_window_beginning_after_a_command_still_sees_its_tail():
    """The warm-up is why: a sequence sampled later must know a command fired.

    Without it, every sampled window would begin with a cold trace and the model
    would learn that commands only ever happen at window starts.
    """
    impulses = np.zeros((1000, 1), dtype=np.uint8)
    impulses[300, 0] = 1
    out = command_features(impulses, np.array([360]), window=20, decay_steps=60)
    assert out[0, 0, 1] == pytest.approx(np.exp(-60 / 60), rel=1e-4)


def test_the_trace_is_derived_rather_than_stored():
    """docs/DECISIONS.md D11: a stored trace costs four times the impulses.

    uint8 impulses against a float32 trace of the same shape.
    """
    impulses = np.zeros((10_000, 11), dtype=np.uint8)
    assert decay(impulses).nbytes == 4 * impulses.nbytes
