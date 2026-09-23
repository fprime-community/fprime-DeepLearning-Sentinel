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

3. **(!) Where the bounds checks are.** This began as a disclosure -- **559 unchecked
   array accesses**, in every module carrying the annotation, while
   `master:docs/DESIGN.md` argued the language case on `[@zero_alloc strict]` and said
   nothing about bounds. A reader could reasonably have concluded the training
   arithmetic was checked. It was not.

   **D82 changed that rather than documenting it.** Measured first: the annotation did
   NOT force the unchecked accesses -- OCaml's bounds-failure path raises a
   preallocated exception and allocates nothing, so `strict` holds either way. The
   flown modules now compile against a bounds-checked accessor, and what is pinned
   below is the split: **502** accesses resolving through an accessor, and **24**
   genuinely unchecked, all of them in drivers that say they are not part of the claim.

`scripts/oxcaml_checked.sh` is the standing measurement: it compiles the annotated
modules as they fly and against the unchecked comparison arm, and both must hold
`[@zero_alloc strict]`. If the flown arm ever stops holding, the checks have become
unaffordable and D82 has to be revisited rather than quietly reverted.
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
EXCLUDED = {"acc.ml", "acc_int.ml", "acc_unchecked.ml"}
STRICT_SITES = 56
ASSUME_SITES = 0

#: (!) D82 SPLIT THIS FIGURE IN TWO, AND THE SPLIT IS THE POINT. The call sites keep
#: spelling `Array.unsafe_get`, deliberately -- a module that opens `Acc` or `Acc_int`
#: resolves that name to the CHECKED operation, so the diff stayed small and the sites
#: stayed greppable. Counting the spelling therefore no longer says what is checked;
#: whether the module opens an accessor does.
CHECKED_THROUGH_ACCESSOR = 502

#: What is genuinely unchecked, and it is only the drivers: `*_check.ml` and
#: `deep_f32_exhaustive.ml`, each of which opens with its own "not annotated and not
#: part of the claim". They are apparatus, they carry no `[@zero_alloc strict]`, and
#: nothing they compute reaches the flown object.
GENUINELY_UNCHECKED = 24


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


def test_every_annotated_module_opens_an_accessor() -> None:
    """(!) D82's actual claim, and the one worth guarding.

    A module carrying `[@zero_alloc strict]` performs the training arithmetic. If it
    does not open an accessor, its `Array.unsafe_get` is the raw primitive and its
    accesses are unchecked -- which is what every one of them was until 2026-09-22.
    """
    missing = []
    for p in _modules():
        text = p.read_text(encoding="utf-8")
        if DEFINITION.search(text) and not re.search(r"^open Acc", text, re.M):
            missing.append(p.name)
    assert not missing, (
        f"these modules carry `[@zero_alloc strict]` and do NOT open an accessor, so "
        f"their element accesses are unchecked: {missing}. D82 made the flown "
        "retrainer bounds-checked; a new annotated module has to opt in the same way.")


def test_the_checked_and_unchecked_counts_are_both_pinned() -> None:
    checked = unchecked = 0
    drivers = {}
    for p in _modules():
        text = p.read_text(encoding="utf-8")
        n = len(UNCHECKED.findall(text))
        if re.search(r"^open Acc", text, re.M):
            checked += n
        else:
            unchecked += n
            if n:
                drivers[p.name] = n
    assert checked == CHECKED_THROUGH_ACCESSOR, (
        f"{checked} accesses resolve through an accessor, pinned at "
        f"{CHECKED_THROUGH_ACCESSOR}. Moving it is fine; moving it silently is not.")
    assert unchecked == GENUINELY_UNCHECKED, (
        f"{unchecked} genuinely unchecked accesses, pinned at {GENUINELY_UNCHECKED}. "
        f"Found in: {drivers}. Every one of these should be in a driver that says it "
        "is not part of the claim -- if a module on the flight path appears here, D82 "
        "has been undone.")
    assert all(n.endswith("_check.ml") or n.endswith("_exhaustive.ml")
               for n in drivers), (
        f"an unchecked access is outside the drivers: {drivers}")
