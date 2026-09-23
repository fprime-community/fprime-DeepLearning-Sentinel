"""(!) The claim the retrainer rests on, re-derived instead of quoted.

`master:docs/DESIGN.md` tells a reader that flight rules forbid heap allocation after
init and that OxCaml's `[@zero_alloc strict]` makes a violation **"a compile error,
transitively across callees"**. That sentence is the whole case for the language, and
until 2026-09-22 the evidence for it was **prose**: `docs/MODELS.md` 47.9's X8 row
quoted one line of compiler output, `scripts/oxcaml_e3.sh` could reproduce it, and
nothing committed, checked or re-ran anything. A claim a reviewer cannot re-run is the
condition D69 was written about, and this one is load-bearing.

Two checks, and the second is why the first is worth having:

1. **The committed rejection is well-formed.** `tests/fixtures/oxcaml_e3_x8_rejection.log`
   is the compiler's own output, captured from the probe, and it has to keep naming
   both halves: the annotation that failed, and the call that made it fail. This runs
   on any tree, with no toolchain.

2. **It is still true.** With the OxCaml switch present, the probe is re-run and the
   live output compared against the committed one. `scripts/oxcaml_e3.sh` injects
   `ignore (Sys.opaque_identity (Array.make 8 0.0))` into a copy of `zalloc.ml`'s
   `matmul` and **fails if that copy BUILDS** -- its own words are "(!) BUILT -- the
   gate cannot fail. X8 FAILED." So the script carries the negative direction, and
   running it carries the positive one too, because the UNMODIFIED file is compiled
   clean under `-zero-alloc-check all` in the same run (X7/X9).

Without the switch this SKIPS and names the script that would satisfy it, which is
D79: a build-dependent gate skips loudly and never inside a green pass.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "oxcaml_e3_x8_rejection.log"
if not FIXTURE.exists():                      # the file lives beside this one
    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "oxcaml_e3_x8_rejection.log"

PROBE = ROOT / "scripts" / "oxcaml_e3.sh"
SWITCH_OCAMLOPT = ROOT / "oxcaml" / ".opam" / "5.2.0+ox" / "bin" / "ocamlopt"

#: Both halves of the rejection. The first names the annotation that failed and the
#: function it was on; the second names the call that allocated. `docs/MODELS.md`
#: 47.9 X8 quotes only the second, which is why the first was never guarded.
ANNOTATION_ERROR = "Annotation check for zero_alloc strict failed on function"
ALLOCATION_ERROR = "called function may allocate (external call to caml_array_make)"


def test_the_committed_rejection_names_the_annotation_and_the_call() -> None:
    """No toolchain needed. The evidence has to be evidence."""
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing. It is the compiler's own refusal of a deliberately "
        "allocating function, and it is what `master:docs/DESIGN.md`'s 'a compile "
        "error, transitively across callees' rests on.")
    text = FIXTURE.read_text(encoding="utf-8")
    assert ANNOTATION_ERROR in text, (
        f"the committed rejection no longer names the failed annotation: {FIXTURE}")
    assert ALLOCATION_ERROR in text, (
        f"the committed rejection no longer names the allocating call: {FIXTURE}")
    assert "P_alloc.matmul" in text, (
        "the rejection should name the function the annotation was on, so a reader "
        "can see it was a TRAINING function and not a contrived one")


@pytest.mark.skipif(not SWITCH_OCAMLOPT.exists(),
                    reason="no OxCaml switch; run scripts/oxcaml_setup.sh")
def test_a_deliberately_allocating_function_still_fails_the_build() -> None:
    """Both directions, and the script already carries both.

    `oxcaml_e3.sh` compiles the unmodified `zalloc.ml` clean under
    `-zero-alloc-check all` (X7/X9, the positive direction: the annotation is
    satisfiable) and then compiles a copy with one allocation injected, exiting
    non-zero if that copy builds (X8, the negative direction: the annotation is
    enforceable). A run that exits 0 has established both.
    """
    assert PROBE.exists(), f"{PROBE} is missing"
    result = subprocess.run(["bash", str(PROBE)], cwd=ROOT, capture_output=True,
                            text=True, timeout=900)
    assert result.returncode == 0, (
        "scripts/oxcaml_e3.sh did not pass. If X8 failed, a deliberately allocating "
        f"function BUILT and the guarantee is gone:\n{result.stdout[-4000:]}\n"
        f"{result.stderr[-2000:]}")
    assert ALLOCATION_ERROR in result.stdout, (
        "E3 passed but did not report the rejection this file pins. Either the probe "
        "stopped printing it or the compiler's wording moved; the committed fixture "
        f"{FIXTURE.name} says what it used to be.\n{result.stdout[-2000:]}")

    # And the live wording still matches the committed evidence, so the fixture is a
    # record of something true rather than something that was once true.
    live = _rejection_log()
    if live is not None:
        assert ANNOTATION_ERROR in live and ALLOCATION_ERROR in live, (
            "the probe's own log no longer carries both halves of the rejection; "
            f"the committed fixture {FIXTURE.name} would now be stale:\n{live}")


@pytest.mark.skipif(not SWITCH_OCAMLOPT.exists(),
                    reason="no OxCaml switch; run scripts/oxcaml_setup.sh")
def test_bounds_checking_and_strict_are_not_in_tension() -> None:
    """(!) The measurement that decides what "safe" may be claimed.

    The reasonable guess was that `[@zero_alloc strict]` forced the 559 unchecked
    accesses: `strict` refuses a function whose paths reach an exceptional return,
    and a checked access raises. `scripts/oxcaml_checked.sh` measures it instead of
    guessing, and the guess is refuted -- every annotated module holds `strict`
    against a bounds-checked accessor, because the bounds-failure path raises a
    preallocated exception and does not allocate.

    D82 then took the other choice: the flown modules compile against the checked
    accessor. So the script's arms are now **as-flown-checked** and
    **comparison-unchecked**, and both must hold `[@zero_alloc strict]` at every
    annotated module -- if the flown arm ever stops holding, the checks have become
    unaffordable and D82 has to be revisited rather than quietly reverted.

    The other direction is what stops the comparison being vacuous: a deliberate
    out-of-range read must be CAUGHT under the flown accessor and NOT caught under
    the unchecked one. Two arms that behave identically would mean the checks are
    doing nothing.
    """
    script = ROOT / "scripts" / "oxcaml_checked.sh"
    assert script.exists(), f"{script} is missing"
    result = subprocess.run(["bash", str(script)], cwd=ROOT, capture_output=True,
                            text=True, timeout=1800)
    assert result.returncode == 0, (
        f"the bounds-checked pass failed:\n{result.stdout[-4000:]}\n"
        f"{result.stderr[-2000:]}")
    out = result.stdout
    assert "CAUGHT: index out of bounds" in out, (
        "the checked accessor did not catch an out-of-range read, so it is not "
        f"checking anything:\n{out[-2000:]}")
    # And the flight accessor must NOT catch it -- otherwise the two variants are the
    # same build and the comparison means nothing.
    unchecked_line = next((ln for ln in out.splitlines()
                           if ln.strip().startswith("unchecked") and "8-element" in ln), "")
    assert unchecked_line and "CAUGHT" not in unchecked_line, (
        "the unchecked comparison arm caught the out-of-range read too, so the two "
        "accessors are not distinct and the measurement is vacuous: "
        f"{unchecked_line!r}")


#: D82.1's cost proxy, measured 2026-09-22 on switch 5.2.0+ox. Shape-determined, so
#: host-independent and exactly reproducible: `Deep_f32.run_cycle` at `t_max` = 250.
#: These are COUNTS, not timings -- stop 35 is kept and the wall clock belongs to the
#: flight-hardware session.
CHECKS_ONE_STEP = 95_212_816
CHECKS_PER_STEP = 95_062_092
CHECKS_FIXED = 150_724


@pytest.mark.skipif(not SWITCH_OCAMLOPT.exists(),
                    reason="no OxCaml switch; run scripts/oxcaml_setup.sh")
def test_the_bounds_check_cost_is_counted_and_pinned() -> None:
    """(!) What D82's bounds checks cost, in a currency stop 35 permits.

    D82 c.5 left the run-time cost unquantified because stop 35 forbids quoting a
    timing figure from this work, and D82.1 kept stop 35 rather than lifting it for a
    laptop measurement. A COUNT is not a timing figure: it is exact, deterministic,
    host-independent, and it is the same currency as 19.8 F4's 70,080 MAC per tick.

    `scripts/oxcaml_count.sh` also asserts that the counting build still holds
    `[@zero_alloc strict]`, so the proxy does not perturb the property being measured.
    """
    script = ROOT / "scripts" / "oxcaml_count.sh"
    assert script.exists(), f"{script} is missing"
    result = subprocess.run(["bash", str(script)], cwd=ROOT, capture_output=True,
                            text=True, timeout=1800)
    assert result.returncode == 0, (
        f"the bounds-check count failed:\n{result.stdout[-3000:]}\n"
        f"{result.stderr[-1500:]}")
    got = {}
    for label, key in (("one optimiser step", "one"),
                       ("per additional step", "per"),
                       ("fixed set-up per cycle", "fixed")):
        m = re.search(rf"{re.escape(label)}\s+([\d,]+)", result.stdout)
        assert m, f"the runner no longer reports {label!r}:\n{result.stdout[-2000:]}"
        got[key] = int(m.group(1).replace(",", ""))
    assert (got["one"], got["per"], got["fixed"]) == (
        CHECKS_ONE_STEP, CHECKS_PER_STEP, CHECKS_FIXED), (
        f"the bounds-check count moved: {got}, pinned at one={CHECKS_ONE_STEP}, "
        f"per={CHECKS_PER_STEP}, fixed={CHECKS_FIXED}. The count is shape-determined, "
        "so a change means the cycle's shapes or its access pattern moved -- both are "
        "things that should move a constant deliberately, not silently.")


def _rejection_log() -> str | None:
    """The log `oxcaml_e3.sh:33` writes, if the run left one behind."""
    log = ROOT / "oxcaml" / "_build" / "e3" / "probes" / "p_alloc.log"
    return log.read_text(encoding="utf-8") if log.exists() else None
