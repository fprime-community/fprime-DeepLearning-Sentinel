"""The operations budget, enforced where it is spent.

docs/DATA.md sets a hard ceiling of 50,000 Class A and 50,000 Class B operations
per calendar month, with a tripwire at 1,000. Neither was enforced anywhere.
``OPS_TRIPWIRE`` and ``OPS_CEILING`` had exactly two consumers: a pre-flight
projection in the ingest CLI, which compares an estimate made *before* a run, and
``OpCounter.summary()``, which formats percentages *after* it. ``OpCounter.record``
raises only on LIST. So this module supplies the missing middle.

It counts through ``sentinel_data.r2``'s existing ``before-send`` hook, which
fires once per HTTP attempt and therefore counts automatic retries the way
Cloudflare bills them. A wrapper counting logical calls would under-report.

**Two thresholds, two different meanings.** Conflating them is why the previous
design enforced neither:

    tripwire   1,000, THIS RUN      A script shaped wrongly -- the whole point of
                                    the manifest is that no run needs hundreds of
                                    operations. Stops and asks for a human, who
                                    may acknowledge it and continue.

    ceiling   50,000, THIS MONTH    The billing limit, counted across every run in
                                    the calendar month, seeded from the ledger.
                                    No override exists.

**The ledger is read-modify-write.** ``upload.publish_ledger`` calls
``new_ledger()`` and would erase the month's history on every run;
``r2.fetch_ledger`` existed but nothing called it. :func:`load` reads the current
month, :func:`commit` adds this run to what it read. Single-writer is assumed --
this project has one -- and the read happens once, at run start, so a run costs
one Class B for the ledger rather than two.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field

from sentinel_data import config as C
from sentinel_data import r2

from .errors import OpsCeilingExceeded, OpsTripwire


@dataclass
class Budget(r2.OpCounter):
    """An :class:`~sentinel_data.r2.OpCounter` that refuses to overspend.

    ``prior_*`` seed the calendar month's already-recorded spend, so the ceiling
    is a real monthly limit rather than a per-process one.
    """

    prior_class_a: int = 0
    prior_class_b: int = 0
    tripwire: int = C.OPS_TRIPWIRE
    acknowledge_tripwire: bool = False
    tripwire_hit: bool = field(default=False, repr=False)

    # -- the enforcement point ---------------------------------------------
    def record(self, operation: str) -> None:
        super().record(operation)   # raises first on any LIST-class operation
        self._enforce()

    def _enforce(self) -> None:
        for cls, month, ceiling in (
            ("Class A", self.month_class_a, C.OPS_CEILING["class_a"]),
            ("Class B", self.month_class_b, C.OPS_CEILING["class_b"]),
        ):
            if month > ceiling:
                raise OpsCeilingExceeded(
                    f"{cls} ceiling exceeded: {month:,} of {ceiling:,} for "
                    f"{r2.current_month()}. This is a hard limit with no override. "
                    f"Nothing further may be read or written until the month rolls over."
                )
        run_peak = max(self.class_a, self.class_b)
        if run_peak >= self.tripwire:
            self.tripwire_hit = True
            if not self.acknowledge_tripwire:
                raise OpsTripwire(
                    f"tripwire: this run has spent {run_peak:,} operations "
                    f"(limit {self.tripwire:,}).\n"
                    f"  {self.describe()}\n"
                    "A correctly shaped run resolves every key from the manifest and "
                    "costs single or low double digits. Fix the shape, or re-run with "
                    "--acknowledge-tripwire if this really is intended."
                )

    # -- reporting ---------------------------------------------------------
    @property
    def month_class_a(self) -> int:
        return self.prior_class_a + self.class_a

    @property
    def month_class_b(self) -> int:
        return self.prior_class_b + self.class_b

    def describe(self) -> str:
        return (
            f"this run: {self.class_a:,} Class A, {self.class_b:,} Class B; "
            f"month {r2.current_month()}: {self.month_class_a:,} / "
            f"{self.month_class_b:,}"
        )

    def report(self) -> str:
        """The block every run prints, measured rather than estimated."""
        lines = ["  OPERATIONS"]
        for cls, run, month, ceiling in (
            ("Class A", self.class_a, self.month_class_a, C.OPS_CEILING["class_a"]),
            ("Class B", self.class_b, self.month_class_b, C.OPS_CEILING["class_b"]),
        ):
            lines.append(
                f"    {cls}  this run {run:>6,}   month {month:>7,} / {ceiling:,}"
                f"  ({100 * month / ceiling:5.2f}%)   [tripwire {self.tripwire:,}/run]"
            )
        if self.by_operation:
            detail = ", ".join(f"{op} x{n}" for op, n in sorted(self.by_operation.items()))
            lines.append(f"    calls: {detail}")
        if self.unclassified:
            lines.append(f"    UNCLASSIFIED: {self.unclassified}")
        if self.tripwire_hit:
            lines.append("    TRIPWIRE ACKNOWLEDGED -- this run was allowed past the limit")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """Provenance for the scorecard. Measured counts, never projections."""
        return {
            "class_a": self.class_a,
            "class_b": self.class_b,
            "month": r2.current_month(),
            "month_class_a": self.month_class_a,
            "month_class_b": self.month_class_b,
            "tripwire": self.tripwire,
            "tripwire_hit": self.tripwire_hit,
            "by_operation": dict(sorted(self.by_operation.items())),
        }


# --------------------------------------------------------------------------
# Ledger: read once at run start, write once at run end
# --------------------------------------------------------------------------
@dataclass
class Ledger:
    """The month-keyed ledger as read at run start, plus what it said."""

    document: dict
    month_class_a: int
    month_class_b: int
    existed: bool


def load(client, bucket: str) -> Ledger:
    """Read `_manifest/ops_ledger.json`. One Class B if it exists."""
    document = r2.fetch_ledger(client, bucket)
    month = document.get("months", {}).get(r2.current_month(), {})
    return Ledger(
        document=document,
        month_class_a=int(month.get("class_a", 0)),
        month_class_b=int(month.get("class_b", 0)),
        existed=bool(document.get("months")),
    )


def commit(client, bucket: str, ledger: Ledger, budget: Budget, log=print) -> dict:
    """Add this run to the ledger it was seeded from, and write it back.

    Costs one Class A, which is counted: the PutObject below has not happened
    when the arithmetic is done, so it is added explicitly rather than left to
    be discovered as a discrepancy next month.
    """
    document = r2.roll_and_add(ledger.document, budget.class_a + 1, budget.class_b)
    blob = json.dumps(document, indent=2).encode() + b"\n"
    r2.put_bytes(
        client, bucket, C.LEDGER_KEY, blob,
        base64.b64encode(hashlib.md5(blob).digest()).decode(),
    )
    log(f"    ledger updated: {C.LEDGER_KEY} ({r2.current_month()})")
    return document


def connect(cfg, *, prior: Ledger | None = None, acknowledge_tripwire: bool = False):
    """An R2 client wired to a Budget. The only way this package reaches R2."""
    budget = Budget(
        prior_class_a=prior.month_class_a if prior else 0,
        prior_class_b=prior.month_class_b if prior else 0,
        acknowledge_tripwire=acknowledge_tripwire,
    )
    client, _ = r2.make_client(cfg, counter=budget)
    return client, budget
