"""D85: the human's ONE command -- approve a certified candidate, uplink it, reload it.

`Objective.md` 11 rule 1: retraining is explicit and human-approved. D85 automates
everything else, so the approval is exactly one command, run by a person who has
read the gate's report:

    PYTHONPATH=src .venv/bin/python -m sentinel_toolkit approve --report R.json \\
        --candidate cand.bin --event-log <the detector's event log>

It REFUSES, and sends nothing, unless:

* the report's verdict is CERTIFY (a REFUSE report can never be approved -- stop 59);
* the candidate file's sha256 is the one the report certified, so what is uplinked is
  what was judged, byte for byte.

Then it runs the report's two commands -- the file uplink and RELOAD_MODEL -- and
reads the DETECTOR'S OWN event log for `ModelReloadAccepted` naming the uplinked
path. **`fprime-cli`'s exit status is never the evidence**: it exits 0 on an unknown
command name (`docs/MODELS.md` 77.8). A refusal logged by the detector -- a CRC, a
width -- is reported as a refusal, with the rollback the detector performs.

`--dry-run` does every check and prints the commands without sending them.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

from .errors import ToolkitError


def check(report: dict, candidate: Path) -> None:
    if report.get("verdict") != "CERTIFY":
        raise ToolkitError(
            f"the report's verdict is {report.get('verdict')!r}: only a CERTIFY report can be "
            f"approved (reasons: {report.get('reasons')})")
    want = report.get("numbers", {}).get("candidate_sha256")
    have = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if want != have:
        raise ToolkitError(
            f"{candidate} is not the candidate the gate certified (sha256 {have}, report "
            f"{want}). Refused: what is uplinked must be what was judged.")
    if len(report.get("commands", [])) != 2:
        raise ToolkitError("a CERTIFY report must carry exactly the uplink and the reload")


def wait_for_reload(event_log: Path, dest: str, since: int, timeout: float) -> str:
    """Poll the detector's event log (from byte `since`) for the reload's outcome."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if event_log.exists():
            text = event_log.read_bytes()[since:].decode("utf-8", "replace")
            for line in text.splitlines():
                if "ModelReloadAccepted" in line and dest in line:
                    return "ACCEPTED"
                if ("ModelReloadRolledBack" in line or "ModelReloadWidthRefused" in line) and dest in line:
                    return "REFUSED: " + line.strip()
        time.sleep(0.5)
    return "NO ANSWER"


def approve(report_path: Path, candidate: Path, event_log: Path | None, *,
            dry_run: bool = False, timeout: float = 60.0, log=print) -> int:
    report = json.loads(report_path.read_text())
    check(report, candidate)
    dest = report["numbers"]["uplink_destination"]
    log(f"  certified   sha256 {report['numbers']['candidate_sha256']}")
    log(f"  part (i)    {report['numbers']['part_i_improvement']:.4%} against m "
        f"{report['numbers']['margin_m']:.4%}")
    for command in report["commands"]:
        log(f"  {'would run' if dry_run else 'running'}  {command}")
    if dry_run:
        return 0
    if event_log is None:
        raise ToolkitError("--event-log is required: the detector's own log is the evidence")
    since = event_log.stat().st_size if event_log.exists() else 0
    for command in report["commands"]:
        out = subprocess.run(shlex.split(command), capture_output=True, text=True)
        text = (out.stdout + out.stderr).strip()
        if "not a known command" in text or out.returncode != 0:
            raise ToolkitError(f"{command.split()[1]} failed: {text[-400:]}")
    verdict = wait_for_reload(event_log, dest, since, timeout)
    log(f"  detector    {verdict}")
    return 0 if verdict == "ACCEPTED" else 1
