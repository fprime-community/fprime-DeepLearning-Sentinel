"""Results: what was measured, over what, and everything needed to repeat it.

Two rules shape this module.

**Every count keeps its denominator.** Scores are stored as
:class:`~sentinel_eval.metrics.counts.Count` objects and rendered as ``k/n`` with
the resolution ``1/n``, and anything below the underpowered threshold says so. A
table of bare rates invites a reader to treat a one-event difference over eleven
events as a result.

**The qualifier travels with the number.** Results get screenshotted and pasted
into messages without the document that explains them, so the single-spacecraft
limitation is written into the scorecard itself and into every `RESULTS.md` row,
not left in `docs/HARNESS.md` to be read by whoever thinks to look.

Provenance carries the manifest revision, the exact channel set, the split, the
git commit, the seeds and the **measured** operation counts -- a floor nobody can
reproduce is not a floor.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field

from . import HARNESS_VERSION
from .metrics.counts import Count
from .metrics.eventwise import EventScore
from .metrics.falsealarm import FalseAlarmScore


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class FoldResult:
    fold: int
    window: tuple[int, int]
    threshold: float | None
    events: EventScore | None
    false_alarms: FalseAlarmScore
    vus_pr: float | None
    vus_detail: dict
    oracle_f_beta: float | None
    oracle_threshold: float | None

    def as_dict(self) -> dict:
        return {
            "fold": self.fold,
            "window": list(self.window),
            "threshold": self.threshold,
            "events": self.events.as_dict() if self.events else None,
            "false_alarms": self.false_alarms.as_dict(),
            "vus_pr": self.vus_pr,
            "vus_detail": self.vus_detail,
            "oracle_best_f0.5": self.oracle_f_beta,
            "oracle_threshold": self.oracle_threshold,
        }


@dataclass
class Scorecard:
    """One detector on one task. Folds, plus the pooled figures."""

    detector: str
    detector_params: dict
    fingerprint: str
    folds: list[FoldResult] = field(default_factory=list)
    pooled: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "detector": self.detector,
            "params": self.detector_params,
            "fingerprint": self.fingerprint,
            "pooled": {k: (v.as_dict() if isinstance(v, Count) else v)
                       for k, v in self.pooled.items()},
            "folds": [f.as_dict() for f in self.folds],
        }


@dataclass
class GateRecord:
    """Every paired set scored in one invocation. The artifact written to `runs/`.

    ``partial`` marks a run that scored only some of its pair. Partial artifacts
    are barred from `docs/RESULTS.md`, which makes single-set publication
    structurally impossible rather than merely discouraged.
    """

    records: list["RunRecord"] = field(default_factory=list)
    partial: bool = False

    def as_dict(self) -> dict:
        return {"partial": self.partial,
                "sets": [r.task["id"] for r in self.records],
                "results": [r.as_dict() for r in self.records]}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)

    def render(self) -> str:
        out = [r.render() for r in self.records]
        if self.partial:
            out.append("\n  PARTIAL RUN -- not all paired sets were scored. "
                       "This artifact is barred from docs/RESULTS.md.")
        return "\n".join(out)

    def for_detector(self, fingerprint: str) -> dict:
        payload = self.as_dict()
        for result in payload["results"]:
            result["scorecards"] = [s for s in result["scorecards"]
                                    if s["fingerprint"] == fingerprint]
        return payload


@dataclass
class RunRecord:
    """Everything one invocation produced. The artifact written to `runs/`."""

    task: dict
    bundle: dict
    coverage: list[dict]
    scorecards: list[Scorecard] = field(default_factory=list)
    ops: dict = field(default_factory=dict)
    limitation: str = ""
    harness_version: str = HARNESS_VERSION
    git_commit: str = field(default_factory=git_commit)

    def as_dict(self) -> dict:
        return {
            "harness_version": self.harness_version,
            "git_commit": self.git_commit,
            "task": self.task,
            "limitation": self.limitation,
            "bundle": self.bundle,
            "coverage": self.coverage,
            "operations": self.ops,
            "scorecards": [s.as_dict() for s in self.scorecards],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=False, default=str)

    # -- rendering ---------------------------------------------------------
    def render(self) -> str:
        task, data = self.task, self.bundle
        lines = [
            "",
            "=" * 78,
            f"  TASK {task.get('id', '?')}   {task.get('role', '')}"
            f"   split {task.get('split', '?')}{task.get('split_params', '')}",
            "=" * 78,
        ]
        if data:
            lines += [
                f"  data     {data.get('mission', '?')} "
                f"{len(data.get('channels', []))} channels {data.get('channels', [])}",
                f"           groups {data.get('groups', [])}   "
                f"{data.get('grid_steps', 0):,} timesteps of "
                f"{data.get('grid_period_seconds', 0)}s   "
                f"normalisation {data.get('normalisation', '?')}",
                f"  manifest {data.get('manifest_generated_utc', '?')} "
                f"(schema {data.get('manifest_schema_version', '?')})   "
                f"commit {self.git_commit}   harness {self.harness_version}",
            ]
        if self.limitation:
            lines += ["", f"  LIMITATION  {self.limitation}"]

        for card in self.scorecards:
            lines += ["", "-" * 78,
                      f"  DETECTOR  {card.detector}  {card.detector_params or ''}"
                      f"   [{card.fingerprint}]", "-" * 78]
            lines += self._render_scores(card)
        if self.ops:
            lines += ["", self.ops.get("report", "")]
        return "\n".join(lines)

    def _render_scores(self, card: Scorecard) -> list[str]:
        """F0.5 leads; recall and precision appear only as its components.

        Recall alone is satisfiable by carpet-bombing -- the trivial baseline
        reached 29/31 by firing 5,535 alarms for 42 events. A detector that
        fires constantly detects everything and is worth nothing, so recall is a
        necessary component and never a headline. Nothing here renders a recall
        without the precision it must be read against.
        """
        pooled = card.pooled
        out: list[str] = []

        if pooled.get("recall") == "disabled for this task by design":
            out.append("    recall DISABLED for this task by design -- see the task note")
        else:
            gate = pooled.get("event_f0.5")
            out.append(f"    GATE      event-wise F0.5"
                       f"{'' if gate is None else f'{gate:>27.3f}'}"
                       f"{'  (undefined)' if gate is None else ''}")
            for label, key in (("from recall", "event_recall"),
                               ("and precision", "event_precision")):
                count = pooled.get(key)
                if isinstance(count, Count):
                    out.append(f"                {label:<20} {count.render()}")

        alarms = pooled.get("rare_event_false_alarms")
        if isinstance(alarms, Count):
            out.append(f"    ADOPTION  rare-event false alarms  {alarms.render()}")
        rate = pooled.get("alarms_per_1000_nominal_timesteps")
        if rate is not None:
            out.append(f"              alarms / 1,000 nominal timesteps {rate:>10.3f}")

        for label, key in (("VUS-PR", "vus_pr"),
                           ("oracle best F0.5 (upper bound)", "oracle_best_f0.5")):
            value = pooled.get(key)
            if value is not None:
                out.append(f"    SUPPORT   {label:<32} {value:>7.3f}")

        components = [("contextual", pooled.get("contextual_recall")),
                      ("headline cell (MVGS)", pooled.get("headline_cell_recall")),
                      ("point", pooled.get("point_recall"))]
        components = [(name, c) for name, c in components if isinstance(c, Count)]
        if components:
            precision = pooled.get("event_precision")
            against = precision.brief() if isinstance(precision, Count) else "n/a"
            out.append(f"    recall components -- read against precision {against}; never alone")
            for name, count in components:
                out.append(f"      {name:<38} {count.render()}")
            for cell, count in sorted((pooled.get("recall_by_cell") or {}).items()):
                out.append(f"        {cell:<36} {count.render()}")

        if len(card.folds) > 1:
            out.append("    per fold:")
            for fold in card.folds:
                value = fold.events.f_beta if fold.events else None
                shown = "n/a" if value is None else f"{value:.3f}"
                recall = fold.events.recall.render(5) if fold.events else ""
                out.append(f"      fold {fold.fold}  F0.5 {shown:<8} from recall {recall}")
        return out
