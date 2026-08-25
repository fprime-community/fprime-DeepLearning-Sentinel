"""Benchmark tasks: what is scored, on which channels, split which way.

Configuration lives here as frozen Python dataclasses rather than in YAML or
TOML. The repository has no config files -- `sentinel_data/config.py` is Python
constants -- so this matches the house style, stays typed and greppable, adds no
dependency, and hashes cleanly into a scorecard's provenance.

**Roles are enforced, not advisory.** `m2-ss1` deduplicates to 18 anomalies with
one to three of them on the test side, which cannot gate anything; it also
carries roughly 600 rare nominal events, which makes it the best false-alarm set
in the dataset by a wide margin. So it is registered with recall **disabled**,
and asking for a recall number from it is an error rather than a footnote.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import TaskError

RECALL = "recall"
ADOPTION = "adoption"


@dataclass(frozen=True)
class Selection:
    """Which channels, resolved against the manifest rather than hard-coded."""

    mission: str
    subsystem: str | None = None
    groups: tuple[int, ...] | None = None
    channel_ids: tuple[str, ...] | None = None
    target_only: bool = True

    def resolve(self, catalog) -> list[str]:
        channels = catalog.channels(
            self.mission,
            subsystem=self.subsystem,
            groups=list(self.groups) if self.groups else None,
            channel_ids=list(self.channel_ids) if self.channel_ids else None,
            target_only=self.target_only and self.channel_ids is None,
            numeric_only=self.channel_ids is None,
        )
        if not channels:
            raise TaskError(f"selection matched no channels: {self}")
        return [c.channel_id for c in channels]


@dataclass(frozen=True)
class Task:
    """One scoring configuration. ``id`` is stable and appears in every artifact."""

    id: str
    selection: Selection
    split: str = "chronological"
    split_params: tuple[tuple[str, float | int], ...] = (("fraction", 0.25),)
    role: str = RECALL
    scores_recall: bool = True
    persistence: int = 1
    headline: str = ""
    note: str = ""
    limitation: str = ""

    @property
    def mission(self) -> str:
        return self.selection.mission

    @property
    def split_kwargs(self) -> dict:
        return dict(self.split_params)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "mission": self.mission, "split": self.split,
            "split_params": self.split_kwargs, "role": self.role,
            "scores_recall": self.scores_recall, "persistence": self.persistence,
            "limitation": self.limitation,
        }


#: The single-spacecraft caveat travels with every result, because results tables
#: get screenshotted and read without the document that qualifies them.
SINGLE_SPACECRAFT = (
    "Recall rests on Mission1 alone: ESA-ADB holds no second viable recall set "
    "(Mission2 dedupes to 18 anomalies, 1-3 test-side; Mission3 has 8 anomalies "
    "and 4 of 48 channels numeric)."
)

#: Sets reported together, always. `m1-ss5` is a strict subset of `m1-g8.9.10`,
#: so one 12-channel load scores both -- reporting the demoted set forever costs
#: 15 Class B rather than the 25 two separate runs would.
#:
#: We changed the evaluation set after seeing a result we did not like. However
#: sound the reasoning, that is externally indistinguishable from cherry-picking,
#: and the claim that the choice was made a priori is forfeit. Reporting both
#: forever is what converts a suspicious edit into a stated scope decision: you
#: cannot cherry-pick if you never discard anything.
PAIRED: dict[str, tuple[str, ...]] = {
    "m1-g8.9.10": ("m1-g8.9.10", "m1-ss5"),
    "m1-ss5": ("m1-g8.9.10", "m1-ss5"),
    "m1-ss5-cv3": ("m1-g8.9.10", "m1-ss5-cv3"),
}

TASKS: dict[str, Task] = {}


def _register(task: Task) -> Task:
    TASKS[task.id] = task
    return task


_register(Task(
    id="m1-ss5",
    selection=Selection("mission1", subsystem="subsystem_5"),
    split="chronological",
    split_params=(("fraction", 0.25),),
    headline="point-anomaly coverage and fast iteration",
    note=("6 channels, one group, all target, perfectly synchronised, and the only channel "
          "set in ESA-ADB containing point anomalies -- all 11. DEMOTED from primary after "
          "the baseline run: 18 of its 38 headline-cell events register as sub-grid-cell "
          "spikes, the only group in mission1 where that happens, so it selects for the "
          "fast regime a per-channel moving average already handles. Retained and reported "
          "alongside m1-g8.9.10 in every result, never discarded."),
    limitation=SINGLE_SPACECRAFT,
))

_register(Task(
    id="m1-ss5-cv3",
    selection=Selection("mission1", subsystem="subsystem_5"),
    split="forward_chaining",
    split_params=(("seed_fraction", 0.25), ("folds", 3)),
    headline="point-anomaly coverage, folded",
    note=("Same channels as m1-ss5, three forward-chaining folds. Scores 42/51 anomalies, "
          "31/38 headline-cell and 11/11 point, while each fold trains on more history "
          "than the last."),
    limitation=SINGLE_SPACECRAFT,
))

_register(Task(
    id="m1-g8.9.10",
    selection=Selection("mission1", groups=(8, 9, 10)),
    split="forward_chaining",
    split_params=(("seed_fraction", 0.25), ("folds", 3)),
    headline="GATE -- PRIMARY recall set",
    note=("12 channels across groups 8+9+10, one spacecraft, two subsystems. PROMOTED to "
          "primary after the baseline run on footprint evidence measured post-hoc: it is "
          "the same events as m1-ss5 but with their real multi-hour extent -- 0 of 40 "
          "headline-cell events are sub-grid-cell here against 18 of 38 on group 8 alone. "
          "Contains group 8, so all 11 point anomalies are retained."),
    limitation=SINGLE_SPACECRAFT,
))

_register(Task(
    id="m1-g3",
    selection=Selection("mission1", groups=(3,)),
    split="forward_chaining",
    split_params=(("seed_fraction", 0.25), ("folds", 3)),
    headline="HELD BACK -- recall. Run once, at the end, settings frozen",
    note=("8 channels, Mission1 group 3. Nominated as a held-back recall set BEFORE "
          "any decision-layer tuning began (docs/MODELS.md section 5), and untouched "
          "since: no result has ever been computed on it. 14 headline-cell events, "
          "median footprint 3,594 timesteps and ZERO sub-grid-cell events -- the "
          "opposite of the group 8 problem that forced the primary set's promotion. "
          "m1-ss5 cannot serve this purpose: it is a strict subset of m1-g8.9.10 "
          "carrying the same events, already inspected in detail. Run once, with "
          "settings already frozen. If the result disappoints we do not re-tune: "
          "tuning against a held-back set converts it into another training set."),
    limitation=SINGLE_SPACECRAFT,
))

_register(Task(
    id="m2-ss1",
    selection=Selection("mission2", subsystem="subsystem_1", groups=(5, 8, 9, 10)),
    split="chronological",
    split_params=(("fraction", 0.30),),
    role=ADOPTION,
    scores_recall=False,
    headline="GATE -- adoption number only",
    note=("An independent spacecraft, 12 channels, ~600 rare nominal events. Recall is "
          "DISABLED: this set dedupes to 18 anomalies with 1-3 test-side, which cannot "
          "carry a comparison. Also HELD BACK by construction -- nothing has ever been "
          "scored on it, so it is the clean test of whether decision-layer tuning "
          "reduced false alarms or merely fitted the 48 rare events on m1-g8.9.10. "
          "Run once, at the end, settings frozen. See docs/MODELS.md section 5."),
    limitation="Recall is not defined on this task by design; see note.",
))

_register(Task(
    id="synthetic",
    selection=Selection("missionX", target_only=False),
    split="forward_chaining",
    split_params=(("seed_fraction", 0.25), ("folds", 3)),
    headline="offline fixture -- zero R2 operations",
    note="Generated data. Used by the test suite and for development.",
    limitation="Synthetic. Never a result.",
))


def get(task_id: str) -> Task:
    try:
        return TASKS[task_id]
    except KeyError:
        raise TaskError(
            f"unknown task {task_id!r}; available: {', '.join(sorted(TASKS))}"
        ) from None


def paired_with(task_id: str) -> tuple[str, ...]:
    """Every task that must be reported whenever ``task_id`` is."""
    return PAIRED.get(task_id, (task_id,))


def listing() -> str:
    rows = []
    for task in sorted(TASKS.values(), key=lambda t: t.id):
        flag = "" if task.scores_recall else "   [recall disabled]"
        rows.append(f"  {task.id:<14} {task.role:<9} {task.headline}{flag}")
    return "\n".join(rows)
