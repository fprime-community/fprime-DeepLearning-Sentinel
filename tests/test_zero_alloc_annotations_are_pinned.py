"""The annotation count, the absence of `assume`, and the unchecked-access count.

Three figures the OxCaml case rests on, and **none of them was guarded by anything
under `tests/`** until 2026-09-22: no test in this directory mentioned `zero_alloc` at
all. They were checked only by `scripts/oxcaml_s*.sh`, which need the switch, so on any
tree without it the figures were prose.

1. **56 `[@zero_alloc strict]` annotation sites.** A naive grep for the string gives
   **68**; twelve of those are comments, and one of the twelve
   (`zalloc.ml`) is a complete `let[@zero_alloc strict] ...` quoted INSIDE a comment.
   A guard that counted the string would pin the wrong number, so this counts
   definitions.

2. **Zero `assume`, in any form.** `strict` is the only variant used anywhere. This is
   the figure `master:docs/DESIGN.md` states as "with zero `assume` annotations", and
   an `assume` is how an annotation stops meaning anything -- it asserts the property
   instead of proving it.

3. **(!) 559 unchecked array accesses.** Not a target and not an achievement: a
   disclosure. `master:docs/DESIGN.md` argues the language case on
   `[@zero_alloc strict]` and says nothing about bounds, and a reader could reasonably
   have concluded the training arithmetic was checked. It is not. The number is pinned
   so that it moves deliberately and visibly, in either direction.

`scripts/oxcaml_checked.sh` is the measurement that says what the number COSTS:
against a bounds-checked accessor every annotated module still holds
`[@zero_alloc strict]`, so the accesses are a choice rather than something the
annotation forced.
"""
from __future__ import annotations

import pathlib
import re

RETRAINER = pathlib.Path(__file__).resolve().parents[1] / "oxcaml" / "retrainer"

#: A definition, not the string: `let` or `and` opening a line, then the attribute.
DEFINITION = re.compile(r"^[ \t]*(?:let|and)\[@zero_alloc strict\]", re.MULTILINE)

#: Any spelling of the attribute at all, comments included.
ANY_MENTION = re.compile(r"\[@zero_alloc")

#: The escape hatch. `[@zero_alloc assume]` and its parameterised forms.
ASSUME = re.compile(r"\[@zero_alloc[^\]]*assume")

#: Unchecked element access, `Array` and `Bigarray.Array1` alike.
UNCHECKED = re.compile(r"\bArray1?\.unsafe_(?:get|set)\b")

#: Measured 2026-09-22 on `dev`. The accessor pair `acc.ml`, `acc_checked.ml` and
#: `acc_checked_int.ml` are excluded from the access count: they are the instrument,
#: they contain the primitive by construction, and counting them would make the
#: disclosure drift every time the instrument is touched.
EXCLUDED = {"acc.ml", "acc_checked.ml", "acc_checked_int.ml"}
STRICT_SITES = 56
ASSUME_SITES = 0
UNCHECKED_ACCESSES = 559


def _modules():
    return sorted(p for p in RETRAINER.glob("*.ml") if p.name not in EXCLUDED)


def test_the_strict_annotation_count_is_what_the_record_states() -> None:
    per_file = {p.name: len(DEFINITION.findall(p.read_text(encoding="utf-8")))
                for p in _modules()}
    total = sum(per_file.values())
    assert total == STRICT_SITES, (
        f"{total} `[@zero_alloc strict]` definitions, pinned at {STRICT_SITES}. "
        "If a module gained or lost one legitimately, move this constant and say so "
        f"in the same commit. Per file: "
        f"{ {k: v for k, v in per_file.items() if v} }")


def test_counting_the_string_would_have_given_the_wrong_number() -> None:
    """The positive control for the check above.

    Without this, `STRICT_SITES` could be matched by a regex that happens to count
    comments too, and nothing would say the two differ. They differ by twelve.
    """
    mentions = sum(len(ANY_MENTION.findall(p.read_text(encoding="utf-8")))
                   for p in _modules())
    assert mentions > STRICT_SITES, (
        "the string count no longer exceeds the definition count; either the comments "
        "that mention the attribute were removed, or DEFINITION has started matching "
        "them, and in the second case this guard has gone blind")


def test_there_is_no_assume_anywhere() -> None:
    offenders = {p.name: ASSUME.findall(p.read_text(encoding="utf-8"))
                 for p in _modules()}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders and ASSUME_SITES == 0, (
        f"`[@zero_alloc ... assume]` found: {offenders}. An `assume` asserts the "
        "property instead of proving it, and `master:docs/DESIGN.md` states the "
        "arithmetic holds 'with zero assume annotations'.")


def test_the_unchecked_access_count_is_disclosed_and_pinned() -> None:
    per_file = {p.name: len(UNCHECKED.findall(p.read_text(encoding="utf-8")))
                for p in _modules()}
    total = sum(per_file.values())
    assert total == UNCHECKED_ACCESSES, (
        f"{total} unchecked array accesses, pinned at {UNCHECKED_ACCESSES}. This "
        "number is a disclosure, not a target: it is what stops 'memory safe' being "
        "a claim this tree supports. Moving it DOWN is progress and moving it UP is a "
        "decision -- either way, move the constant deliberately and say which. "
        f"Per file: { {k: v for k, v in per_file.items() if v} }")
