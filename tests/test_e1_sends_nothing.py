"""D85.1 and docs/MODELS.md 78.11: E1 shows the human's step and sends nothing.

D85.1 forbids any shadow model being swapped in, on the testbed or in any deployment,
until a route holds C1 at zero. `scripts/d85_e1.sh` used to run `sentinel_toolkit
approve` for real, which uplinks and commands `RELOAD_MODEL` whenever the gate certifies.
78.11 re-registered E1 as E1-dry: the lines are printed, `approve --dry-run` checks the
report, and nothing reaches the detector.

This reads the script as text. A line EXECUTES unless it is a comment or an `echo`, so:
- every `sentinel_toolkit approve` that executes carries `--dry-run`;
- no executed line runs `fprime-cli file-uplink` or `fprime-cli command-send`.

Both directions: two mutants of the real script, one executing approve without
`--dry-run` and one executing the uplink, must each be caught.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "d85_e1.sh"


def _executed(text: str) -> list[str]:
    """Logical lines that run: continuations joined, comments and echoes dropped."""
    lines, buf = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        lines.append(buf + line)
        buf = ""
    out = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("echo "):
            continue
        out.append(s)
    return out


def violations(text: str) -> list[str]:
    bad = []
    for line in _executed(text):
        if "sentinel_toolkit approve" in line and "--dry-run" not in line:
            bad.append(f"approve without --dry-run: {line}")
        if "fprime-cli file-uplink" in line or "fprime-cli command-send" in line:
            bad.append(f"sends to the detector: {line}")
    return bad


def test_e1_sends_nothing() -> None:
    text = SCRIPT.read_text()
    assert "sentinel_toolkit approve --dry-run" in " ".join(_executed(text)), (
        "E1 no longer exercises approve at all; 78.11's E1d.4 needs the dry run")
    assert violations(text) == []


def test_the_guard_catches_an_approve_that_sends() -> None:
    mutant = SCRIPT.read_text().replace("sentinel_toolkit approve --dry-run",
                                        "sentinel_toolkit approve")
    assert any("approve without --dry-run" in v for v in violations(mutant))


def test_the_guard_catches_an_executed_uplink() -> None:
    mutant = SCRIPT.read_text() + (
        '\nfprime-cli file-uplink "${RUN}/ground/candidate.bin" RetrainApproved.bin\n')
    assert any("sends to the detector" in v for v in violations(mutant))


def test_a_printed_line_is_not_an_executed_one() -> None:
    printed = 'echo "   NOT SENT: fprime-cli command-send X --arguments Y"\n'
    assert violations(printed) == []
