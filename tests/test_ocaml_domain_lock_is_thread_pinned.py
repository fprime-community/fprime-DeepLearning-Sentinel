"""(!) OCaml 5 pins the domain lock to the thread that called `caml_startup`, and
nothing reproduced that until this file.

`docs/MODELS.md` 65.6 records the first real tick of the retrainer inside a deployment
dying with `Fatal error: no domain lock held`, and records the fix:
`fprime/SentinelRef/Retrainer/Retrainer.cpp:97-108` boots the runtime lazily, inside
`schedIn_handler`, on the `Svc.ActiveRateGroup` task that then owns the domain for the
life of the process.

**That fix was believed in one direction only.** The deployment stopped crashing, which
shows the fix is sufficient; nothing showed the failure was real, reproducible, or that
the guard could see it. `docs/MODELS.md` 65.8 already caught one guard in this track that
had only ever passed and was blind. So there are two checks here and they point opposite
ways:

1. **The crash, on demand.** `oxcaml/retrainer/domain_lock_probe.cpp` boots on the main
   thread -- exactly as `configureTopology()` would -- and calls into OCaml from a second
   pthread. It must die, and it must die with that message.
2. **The same calls on the booting thread.** Same binary, same entry point, one thread.
   It must succeed.

And a third, structural, because the first two only test the code as it is today:

3. **Every OCaml entry point is reachable only from `schedIn`.** Route (b) -- boot on the
   ticking thread -- is not self-enforcing. It holds only while `schedIn` is the single
   way in. `docs/MODELS.md` 65's hub work adds a second caller, the socket read task, and
   a port handler wired to it would run there. This asserts by shape what checks 1 and 2
   assert by execution.

Build the probe with `scripts/oxcaml_s65a.sh`. Checks 1 and 2 skip when it is absent,
which on a clone without the OxCaml switch it always is; check 3 always runs.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBE = ROOT / "oxcaml" / "_build" / "s65a" / "domain_lock_probe"
RETRAINER = ROOT / "fprime" / "SentinelRef" / "Retrainer" / "Retrainer.cpp"
RETRAIN_TOP = (ROOT / "fprime" / "SentinelRetrain" / "Top"
               / "SentinelRetrainTopology.cpp")

#: Every C entry point into the OxCaml side. Three surfaces: E1's pipe, the
#: training cycle, and 72's shadow model file. `docs/MODELS.md` 65, 61 and 72
#: respectively.
#:
#: (!) `shadow` WAS ADDED AT SECTION 72 AND THE OMISSION WOULD HAVE BEEN SILENT.
#: Until it was, this guard read a file containing five new OCaml entry points and
#: reported that every entry point was on the ticking thread -- true of the ones it
#: could see, and worthless. That is 65.8's blind symbol check and 7's invisible
#: citation a third time, so the widening is shown to fail first
#: (`test_the_pattern_would_catch_a_shadow_call_off_the_ticking_thread`).
OCAML_ENTRY = re.compile(r"\bsentinel_(?:retrainer|cycle|shadow)_\w+\s*\(")

#: The pattern this replaced, kept so the both-directions test can show what it
#: missed. Never used to judge the source.
OCAML_ENTRY_BEFORE_72 = re.compile(r"\bsentinel_(?:retrainer|cycle)_\w+\s*\(")

#: The only two functions permitted to contain one. `boot` is included because
#: `schedIn_handler` is the only thing that calls it (`Retrainer.cpp:106-108`),
#: so it inherits the ticking thread.
ON_THE_TICKING_THREAD = {"boot", "schedIn_handler"}


def _require_probe():
    if not PROBE.exists():
        pytest.skip(f"{PROBE.name} is not built; run scripts/oxcaml_s65a.sh")


def _run(mode: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(PROBE), mode], cwd=PROBE.parent,
                          capture_output=True, text=True, timeout=120)


def _functions(source: str) -> dict[str, range]:
    """`Retrainer::name` -> the line range of its body, 1-indexed."""
    lines = source.splitlines()
    starts = []
    for i, line in enumerate(lines, start=1):
        match = re.match(r"^[A-Za-z_][\w:&*<>, ]*\bRetrainer::(\w+)\s*\(", line)
        if match:
            starts.append((match.group(1), i))
    spans = {}
    for name, start in starts:
        end = next((j for j in range(start, len(lines) + 1)
                    if lines[j - 1] == "}"), len(lines))
        spans[name] = range(start, end + 1)
    return spans


def test_a_cross_thread_call_into_ocaml_is_fatal() -> None:
    """The negative control. This is the crash `docs/MODELS.md` 65.6 reported."""
    _require_probe()
    proc = _run("cross")
    assert proc.returncode != 0, (
        "the cross-thread call into OCaml returned instead of aborting. The fix at "
        "Retrainer.cpp:97-108 exists because OCaml 5 makes this fatal; if it is not "
        "fatal in this switch, the reason for booting on the ticking thread is gone "
        "and 65.6's account needs re-reading, not this test relaxing.")
    assert proc.returncode != 4, (
        "the probe reached its own no-abort line, which means the call completed on a "
        "thread that does not hold the domain lock.")
    assert "no domain lock held" in (proc.stderr + proc.stdout).lower(), (
        "the cross-thread call died, but not with 'no domain lock held'. Something "
        f"else killed it and this control proves nothing:\n{proc.stderr[-800:]}")


def test_the_same_calls_succeed_on_the_thread_that_booted_the_runtime() -> None:
    """The positive direction. Same binary, same entry point, one thread."""
    _require_probe()
    proc = _run("same")
    assert proc.returncode == 0, (
        "the same-thread call did not succeed, so the cross-thread abort above is not "
        f"evidence about threads at all:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")


def test_every_ocaml_entry_point_sits_on_the_ticking_thread() -> None:
    """(!) Route (b) is not self-enforcing, and the hub adds a second caller.

    Booting on the ticking thread pins the domain there. It keeps the process alive
    only while every call in also happens there. An `async` port handler on a
    `queued` component is dispatched by `dispatchCurrentMessages()` from `schedIn`
    and is therefore fine; a `sync` handler wired to the hub runs on the socket read
    task and is not.
    """
    source = RETRAINER.read_text()
    spans = _functions(source)
    assert ON_THE_TICKING_THREAD <= set(spans), (
        f"Retrainer.cpp no longer defines {sorted(ON_THE_TICKING_THREAD)}; this guard "
        f"reads {sorted(spans)} and has gone stale rather than clean.")

    offenders = {}
    for number, line in enumerate(source.splitlines(), start=1):
        if not OCAML_ENTRY.search(line):
            continue
        holder = next((n for n, span in spans.items() if number in span), None)
        if holder not in ON_THE_TICKING_THREAD:
            offenders[number] = (holder, line.strip())

    assert not offenders, (
        f"OCaml entry points outside {sorted(ON_THE_TICKING_THREAD)}: {offenders}. "
        "OCaml 5 pins the domain lock to the thread that called caml_startup, which "
        "here is the Svc.ActiveRateGroup task. A call from any other handler aborts "
        "the process with 'no domain lock held' -- docs/MODELS.md 65.6.")


def test_the_retrainer_deployment_does_not_boot_the_runtime_in_topology_setup() -> None:
    """The mistake that produced 65.6's crash, kept out by assertion."""
    source = RETRAIN_TOP.read_text()
    calls = [line.strip() for line in source.splitlines()
             if OCAML_ENTRY.search(line) and not line.lstrip().startswith("//")]
    assert not calls, (
        f"SentinelRetrainTopology.cpp calls into OCaml: {calls}. configureTopology() "
        "runs on the topology's main thread and rateGroup_1Hz calls schedIn on its "
        "own task, so a runtime started here aborts on the first tick.")


def test_the_pattern_would_catch_a_shadow_call_off_the_ticking_thread() -> None:
    """(!) BOTH DIRECTIONS, and the wrong direction is the one that matters here.

    Section 72 added `sentinel_shadow_load`, `_write`, `_export` and `_loss`. The
    pattern in force before it matched none of them, so a shadow call placed on any
    other thread would have passed this file's central assertion in silence. The
    probe below is a synthetic source, never the real one: it puts a shadow call in
    a handler that is NOT on the ticking thread and asserts the current pattern
    sees it and the old one does not.
    """
    probe = (
        "void Retrainer::boot() { sentinel_retrainer_boot(); }\n"
        "void Retrainer::hubIn_handler() { sentinel_shadow_load(buf, n); }\n"
    )
    offending = [line for line in probe.splitlines() if OCAML_ENTRY.search(line)
                 and "hubIn_handler" in line]
    assert offending, (
        "the current pattern does not match a shadow entry point; widening it at "
        "Section 72 achieved nothing")

    missed = [line for line in probe.splitlines()
              if OCAML_ENTRY_BEFORE_72.search(line) and "hubIn_handler" in line]
    assert not missed, (
        "the pre-72 pattern already matched shadow entry points, so the widening "
        "was not the fix this docstring claims it was")


def test_the_guard_reads_a_file_that_actually_contains_shadow_entry_points() -> None:
    """A pattern that matches nothing in the real source is not a guard either."""
    source = RETRAINER.read_text()
    found = [m.group(0) for m in OCAML_ENTRY.finditer(source)
             if "shadow" in m.group(0)]
    assert found, (
        "Retrainer.cpp contains no sentinel_shadow_* call, so either Section 72's "
        "HO1 work was reverted or this guard is watching the wrong file.")
