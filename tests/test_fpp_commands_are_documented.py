"""(!) A component's `sdd.md` said it declared no command while its `.fpp` declared one.

`fprime/Sentinel/Monitor/docs/sdd.md` carried, from work item 9 until 2026-09-21:

    **The command ports are not a Sentinel command.** This component declares no
    command of its own and issues none.

By then `fprime/Sentinel/Monitor/Monitor.fpp` had declared `async command
RELOAD_MODEL(modelPath: string size 40)` at `opcode 0x10` since work item 10, and the
`.fpp` comments, `README.md` and `fprime/SentinelRef/README.md` had all been corrected.
Only the `sdd` was missed, and `flight/` and `fprime/` are byte-identical across `dev`
and `master` by tree SHA, so the stale sentence shipped on the customer branch too. It
is the cheapest kind of claim for a reviewer to disprove: open one file.

`Objective.md` 11 rule 3 is about what a component **issues**. "It issues no command and
has no commanding port of any kind" stays true and is not what this file objects to.
"Declares no command of its own" is a different claim, and for `Monitor` it is false.

**Rule 17: a note asking a future reader to remember is not a fix.** This is the guard.

Three checks, and the third is why the first two can be believed:

1. **Every command an `.fpp` declares is named in its component's `sdd.md`.**
2. **No `sdd.md` denies a command its own `.fpp` declares.** Scoped to components that
   actually declare one -- `PowerSim/docs/sdd.md` and `Retrainer/docs/sdd.md` say they
   declare no command of their own and are correct, because they do not.
3. **The parser is not blind.** If the command regex silently matched nothing, checks 1
   and 2 would pass over an empty set and prove nothing. This asserts the known
   ground truth: exactly one command is declared under `fprime/`, `RELOAD_MODEL` in
   `Monitor.fpp`, and it carries opcode `0x10`.

A component with no sdd document has nothing that can go stale and is not checked; the
`Top/` topology files declare no commands and fall out of the walk for the same reason.
`fprime/lib/` is the gitignored framework checkout and is never walked, and neither
is `fprime/fprime-venv/` -- **which this file claimed and did not do until
2026-09-22**. Both are rebuilt by `scripts/fprime_setup.sh`, both are gitignored, and
the exclusion now names both, as `conftest.py` already did. See `NOT_OURS`.
"""
from __future__ import annotations

import pathlib
import re

FPRIME = pathlib.Path(__file__).resolve().parents[1] / "fprime"

#: `async command NAME`, `sync command NAME`, `guarded command NAME` at the start of a
#: line, which is how FPP declares one. The opcode sits on a later line, so it is not
#: part of this pattern.
COMMAND = re.compile(r"^\s*(?:async|sync|guarded)\s+command\s+([A-Za-z_][A-Za-z_0-9]*)",
                     re.MULTILINE)

#: Phrases that deny DECLARING or RECEIVING a command. "Issues no command", "commands
#: nothing" and "no commanding port" are claims about what the component SENDS, which is
#: what Objective.md 11 rule 3 governs, and they stay true -- they are deliberately not
#: in this list.
DENIALS = (
    "declares no command",
    "no command of its own",
    "no command receive port",
    "not a sentinel command",
)


#: (!) BOTH gitignored subtrees, named exactly as `conftest.py` excludes them from
#: collection -- and the second was missing, which made this file's own docstring
#: false. `fprime/lib/` is the framework checkout; `fprime/fprime-venv/` is the tool
#: virtualenv, and it ships F's cookiecutter component template at
#: `lib/python3.14/site-packages/fprime/cookiecutter_templates/`, whose `.fpp`
#: declares `TODO`, `TODO_1` and `TODO_2`. The old test read the FIRST path part
#: only, so `("lib",)` was excluded and `("fprime-venv", "lib", ...)` was not: this
#: walk reported three undocumented commands in a component that does not exist.
#: It could only ever fire on a machine that had run `scripts/fprime_setup.sh`,
#: which is why it passed for as long as the checkout was absent.
NOT_OURS = ("lib", "fprime-venv")


def _components(root: pathlib.Path):
    """Every `.fpp` under `root` declaring a command, with its component's sdd path."""
    out = []
    for fpp in sorted(root.rglob("*.fpp")):
        if fpp.relative_to(root).parts[0] in NOT_OURS:
            continue
        names = COMMAND.findall(fpp.read_text(encoding="utf-8", errors="replace"))
        if names:
            out.append((fpp, names, fpp.parent / "docs" / "sdd.md"))
    return out


def test_every_fpp_command_appears_in_its_component_sdd() -> None:
    missing = []
    for fpp, names, sdd in _components(FPRIME):
        if not sdd.exists():
            continue
        body = sdd.read_text(encoding="utf-8", errors="replace")
        for name in names:
            if name not in body:
                missing.append(f"{sdd}: `{fpp.name}` declares {name}, the sdd never names it")
    assert missing == [], "commands absent from their component's sdd:\n  " + "\n  ".join(missing)


def test_no_sdd_denies_a_command_its_fpp_declares() -> None:
    wrong = []
    for fpp, names, sdd in _components(FPRIME):
        if not sdd.exists():
            continue
        for n, line in enumerate(sdd.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            low = line.lower()
            for phrase in DENIALS:
                if phrase in low:
                    wrong.append(f"{sdd}:{n}: says \"{phrase}\" while {fpp.name} "
                                 f"declares {', '.join(names)}")
    assert wrong == [], ("an sdd denies a command its own .fpp declares:\n  "
                         + "\n  ".join(wrong))


def test_the_command_parser_can_actually_see_a_command() -> None:
    """The positive control. Without it the two checks above pass over an empty set."""
    found = {fpp.name: names for fpp, names, _ in _components(FPRIME)}
    assert found == {"Monitor.fpp": ["RELOAD_MODEL"]}, (
        "the known ground truth under fprime/ is exactly one declared command, "
        f"RELOAD_MODEL in Monitor.fpp; the parser found {found}")
    text = (FPRIME / "Sentinel" / "Monitor" / "Monitor.fpp").read_text(encoding="utf-8")
    assert "opcode 0x10" in text, "RELOAD_MODEL's opcode moved; the sdd records 0x10"
