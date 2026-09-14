"""The testbed's onboard RED limits are the ones its dictionary declares.

`docs/MODELS.md` 42.3 departure 1 requires the limit trip and the warning to
share a time base. F' evaluates telemetry limits on the **ground** -- *"limit
checking is performed by the ground system based on the dictionary definition"*
(F' v4.3.0 `docs/reference/system-functional/telemetry-chan.md:32`) -- and
**exposes no limit constant to C++**, which was verified by reading the autocoded
`PowerSimComponentAc.hpp`. So `PowerPlant.hpp` carries a second copy and
`PowerSim.cpp` evaluates it onboard.

**A second copy is what that departure forbids. A CHECKED second copy is the only
shape available**, so this is the check: the table is re-derived from the FPP on
every run, the way `tests/test_master_documents_are_current.py` re-derives
`master`'s figures from the source that produces them rather than from prose.

It fails loudly in both directions. A limit changed in the FPP and not in the
header is a testbed measuring against a limit nobody declared; a limit changed in
the header and not the FPP is a dictionary the ground would check against
something else.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FPP = ROOT / "fprime/SentinelRef/PowerSim/PowerSim.fpp"
HPP = ROOT / "fprime/SentinelRef/PowerSim/PowerPlant.hpp"

#: `telemetry Name: F32 id N \ low { red X, yellow Y } \ high { red A, yellow B }`
#: The continuations and the comment block between them are why this is not
#: line-anchored: a pattern that reads a wrapped declaration as *no declaration*
#: is the one failure mode a guard may not have.
_TLM = re.compile(
    r"telemetry\s+(\w+)\s*:\s*F32\s+id\s+(\d+)\s*\\?\s*"
    r"low\s*\{\s*red\s+(-?[\d.]+)\s*,\s*yellow\s+(-?[\d.]+)\s*\}\s*\\?\s*"
    r"high\s*\{\s*red\s+(-?[\d.]+)\s*,\s*yellow\s+(-?[\d.]+)\s*\}")

_ROW = re.compile(r"\{\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\}\s*,?\s*//\s*(\w+)")


def _declared() -> dict[str, tuple[float, float]]:
    """`name -> (red_low, red_high)`, from the dictionary."""
    text = FPP.read_text(encoding="utf-8")
    return {m.group(1): (float(m.group(3)), float(m.group(5)))
            for m in _TLM.finditer(text)}


def _compiled() -> dict[str, tuple[float, float]]:
    """`name -> (low, high)`, from `PLANT_RED`."""
    text = HPP.read_text(encoding="utf-8")
    body = text[text.index("PLANT_RED[PLANT_CHANNELS] = {"):text.index("};", text.index("PLANT_RED["))]
    return {m.group(3): (float(m.group(1)), float(m.group(2))) for m in _ROW.finditer(body)}


def test_the_patterns_still_find_something() -> None:
    """(!) A guard that stops matching has gone blind, which is worse than a
    stale number. Both sides must be non-empty before any comparison means
    anything -- the same argument `test_master_documents_are_current.py:288`
    makes for its own patterns."""
    declared, compiled = _declared(), _compiled()
    assert len(declared) == 8, f"FPP: expected 8 limited channels, read {sorted(declared)}"
    assert len(compiled) == 8, f"PLANT_RED: expected 8 rows, read {sorted(compiled)}"


def test_every_onboard_red_limit_is_the_one_the_dictionary_declares() -> None:
    declared, compiled = _declared(), _compiled()
    assert set(declared) == set(compiled), (
        "the dictionary and PLANT_RED name different channels: "
        f"only in FPP {sorted(set(declared) - set(compiled))}, "
        f"only in PLANT_RED {sorted(set(compiled) - set(declared))}")
    wrong = {n: (declared[n], compiled[n]) for n in declared if declared[n] != compiled[n]}
    assert not wrong, (
        "PowerPlant.hpp's PLANT_RED disagrees with PowerSim.fpp's dictionary, so the "
        f"testbed would measure against a limit nobody declared: {wrong}")


@pytest.mark.parametrize("name", sorted(_declared()))
def test_the_yellow_band_sits_inside_the_red_band(name: str) -> None:
    """Yellow inside red, or the dictionary is not a ladder.

    (!) AND YELLOW IS THE BAR THAT MATTERS, WHICH BUILDING THIS FOUND.
    `docs/MODELS.md` 42.4's T3 was written against the first RED crossing. On the
    seeded run `CellTemp` crosses **yellow at 19,225 and red at 20,140**, so a
    limit check with yellow alarms sees the fault **915 ticks earlier** than the
    red trip. Sentinel has to beat the yellow to beat a limit check at all.
    Recorded at 42.8.
    """
    text = FPP.read_text(encoding="utf-8")
    m = next(x for x in _TLM.finditer(text) if x.group(1) == name)
    red_lo, yel_lo, red_hi, yel_hi = (float(m.group(i)) for i in (3, 4, 5, 6))
    assert red_lo <= yel_lo, f"{name}: yellow low {yel_lo} is outside red low {red_lo}"
    assert yel_hi <= red_hi, f"{name}: yellow high {yel_hi} is outside red high {red_hi}"
