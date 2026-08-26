"""Telecommands, put on the grid as impulses. The other half of telemanom's input.

Hundman et al. feed the LSTM "prior telemetry values for a given channel **and
encoded command information sent to the spacecraft**", and their Figure 3 shows
that encoding letting the model predict a commanded event so that it is not
flagged. Work item 4 reproduced the method without it, which is
`docs/DECISIONS.md` D6 and the reason the false-alarm rate is what it is: our
false alarms are on commanded manoeuvres, and we withheld the one input that
makes them predictable.

**A command is an instant, not a level.** Everything else the harness puts on the
grid is a measurement that persists until the next one, so `grid.resample_zoh`
holds it forward. Holding a command forward would assert that the spacecraft is
being commanded continuously from the first execution to the end of the mission.
So impulses are rasterised, not held, and a grid cell carries the *number* of
executions that fell inside it.

**This module stores impulses and nothing else.** A short exponentially decaying
"recently commanded" trace is what lets a model learn the *shape* of a response
rather than its instant -- but it is a *feature*, not a rasterisation, so it lives
in `sentinel_models.windows` where the model that wants it can derive it per
window. Two reasons, and either alone would decide it: over 14.7M timesteps and
eleven commands a stored float32 decay costs 648 MB against the impulses' 162 MB
(`docs/DECISIONS.md` D11), and the harness has no business deciding what shape a
model's inputs take.

**Commands are not labels.** `sentinel_eval.detector.Context` promises a detector
no labels, ever, and that promise is intact: a telecommand is an input the
spacecraft itself has, known before it is executed, and available in flight.
"""
from __future__ import annotations

import numpy as np

from .errors import TaskError
from .grid import Grid
from .read import Executions

#: A cell holds a count, not a flag, because two executions in one 30-second cell
#: is a different event from one. Saturates rather than wrapping: 255 commands in
#: a cell is already far outside anything the model needs to distinguish.
COUNT_CEILING = 255

def rasterise(executions: Executions, grid: Grid, order: list[str]) -> np.ndarray:
    """Executions onto the grid as ``uint8[T, K]`` counts, columns in ``order``.

    An execution outside the grid's window is **dropped, not clamped**. Clamping
    would pile every pre-grid command onto timestep zero and invent a burst that
    never happened, at exactly the point where a model has no history to judge it
    against.
    """
    if not order:
        raise TaskError("no telecommands selected; pass the ids to rasterise")

    column_of = {name: i for i, name in enumerate(order)}
    out = np.zeros((len(grid), len(order)), dtype=np.uint8)

    inside = (executions.t_anon >= grid.start) & (executions.t_anon < grid.end)
    times = executions.t_anon[inside]
    ids = executions.telecommand_id[inside]

    steps = ((times - grid.start) // grid.period).astype(np.int64)
    for name, column in column_of.items():
        hit = steps[ids == name]
        if hit.size == 0:
            continue
        counts = np.bincount(hit, minlength=len(grid))
        np.minimum(counts, COUNT_CEILING, out=counts)
        out[:, column] = counts.astype(np.uint8)
    return out


def select(descriptions, mission: str, priority: int, executed: set[str]) -> list[str]:
    """The telecommands to model: a priority grade, minus the ones that never fire.

    ESA grades telecommands 0 to 3 and feeds only priority 3 to its own
    Telemanom-ESA and DC-VAE-ESA baselines. **Mission1 holds 11 of those, not the
    15 quoted dataset-wide** -- Mission2 holds the other 4 (`docs/DECISIONS.md`
    D7).

    A command that never executes is dropped. It would be a constant-zero input
    column: harmless to the arithmetic, and it dilutes the command features and
    weakens the ablation it is supposed to inform. 17 of Mission1's 698
    telecommands are declared and never executed.
    """
    rows = descriptions.to_pylist() if hasattr(descriptions, "to_pylist") else descriptions
    graded = [str(r["Telecommand"]) for r in rows
              if str(r["mission"]) == mission and int(r["Priority"]) == priority]
    if not graded:
        raise TaskError(
            f"{mission} has no priority-{priority} telecommands; "
            f"the description table carries mission, Telecommand, Priority"
        )
    return sorted(set(graded) & executed)
