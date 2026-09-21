"""(!) `make -C flight lint` printed "lint: clean" while two of its three configs
had not run.

Decision 79. The target holds the flight core to three clang-tidy configurations:
its own `flight/.clang-tidy`, then F's root config, then F's `release.clang-tidy`,
the one F' reserves for flight code. The last two come from the framework checkout,
which is gitignored. From 2026-09-21, when the build trees were deleted, that
checkout was absent -- so one config ran, and the target printed

    -- framework checkout absent; skipping its two configs
      lint: clean

and exited 0. The skip was announced, but announced INSIDE a green result, which is
the form nobody reads. An inspection on 2026-09-21 recorded the gate as passing.

This target has been here before. Its own comment block records that until
2026-09-11 it ran clang-tidy three times, checked no exit status, and echoed
"clean" unconditionally while nine errors stood. **A gate that cannot go red is not
a gate**, and a gate that goes green on a third of its work is the same defect with
better manners.

Most checks here force `TIDY=` on the command line, which takes the no-clang-tidy
branch and exercises the PARTIAL logic in milliseconds without invoking a linter.
One check runs the real target, because the logic being right in a fixture and
right in the gate people actually run are two different claims.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLIGHT = ROOT / "flight"
FRAMEWORK_CONFIG = ROOT / "fprime" / "lib" / "fprime" / ".clang-tidy"


def _lint(*extra: str, allow_partial: bool = False) -> subprocess.CompletedProcess:
    args = ["make", "-C", "flight", "lint", *extra]
    if allow_partial:
        args.append("LINT_ALLOW_PARTIAL=1")
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def test_a_partial_run_never_calls_itself_clean() -> None:
    proc = _lint("TIDY=")
    assert "PARTIAL" in proc.stdout, f"a partial run did not say so:\n{proc.stdout}"
    assert "lint: clean" not in proc.stdout, (
        f"a partial run called itself clean:\n{proc.stdout}")


def test_a_partial_run_exits_non_zero_without_the_opt_in() -> None:
    """Route (i) of Decision 79: an unbuilt tree cannot produce a green lint by
    accident. The opt-in exists so it can be produced on purpose."""
    proc = _lint("TIDY=")
    assert proc.returncode != 0, (
        f"a partial run exited 0 with no opt-in set:\n{proc.stdout}")


def test_the_opt_in_accepts_a_partial_run_and_still_says_partial() -> None:
    proc = _lint("TIDY=", allow_partial=True)
    assert proc.returncode == 0, f"LINT_ALLOW_PARTIAL did not accept it:\n{proc.stdout}"
    assert "PARTIAL" in proc.stdout, (
        f"the opt-in turned the report back into a pass:\n{proc.stdout}")
    assert "lint: clean" not in proc.stdout, proc.stdout


def test_a_partial_run_names_what_was_skipped_and_how_to_get_it() -> None:
    """A skip nobody can act on is the same as a silent one."""
    proc = _lint("TIDY=", allow_partial=True)
    assert "skipped:" in proc.stdout, f"nothing named what was skipped:\n{proc.stdout}"
    assert "of 3 configs ran" in proc.stdout, (
        f"the report does not say how many configs ran:\n{proc.stdout}")

    if not FRAMEWORK_CONFIG.exists():
        real = _lint(allow_partial=True) if shutil.which("clang-tidy") or Path(
            "/opt/homebrew/opt/llvm/bin/clang-tidy").exists() else None
        if real is not None:
            assert "scripts/fprime_setup.sh" in real.stdout, (
                "the framework-absent skip does not name the script that would "
                f"satisfy it:\n{real.stdout}")


def test_the_real_target_reports_partial_while_the_checkout_is_absent() -> None:
    """The gate as it is actually run, not a fixture of it."""
    if FRAMEWORK_CONFIG.exists():
        pytest.skip("the framework checkout is present; a full run is expected here")
    if not (shutil.which("clang-tidy")
            or Path("/opt/homebrew/opt/llvm/bin/clang-tidy").exists()):
        pytest.skip("clang-tidy is not installed; brew install llvm provides it")
    proc = _lint(allow_partial=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "lint: PARTIAL (1 of 3 configs ran)" in proc.stdout, (
        f"the real target did not report 1 of 3:\n{proc.stdout}")
    assert "lint: clean" not in proc.stdout, proc.stdout


def test_clean_is_reachable_only_when_nothing_was_skipped() -> None:
    """Structural, because the full three-config run cannot be demonstrated on a
    tree without the framework checkout and this session may not rebuild it.

    `flight/Makefile` prints `clean` inside the branch guarded by an empty
    `skipped`, and nowhere else. If a second `clean` ever appears outside that
    branch, this fails and someone reads the recipe.
    """
    recipe = (FLIGHT / "Makefile").read_text(encoding="utf-8")
    body = recipe[recipe.index("\nlint:"):]
    body = body[: body.index("\nclean:")]
    cleans = [ln for ln in body.splitlines() if "lint: clean" in ln]
    assert len(cleans) == 1, f"expected exactly one `lint: clean`, found {cleans}"
    guard = 'if [ -z "$$skipped" ]; then'
    assert guard in body, "the `clean` branch is no longer guarded on an empty skip list"
    assert body.index(guard) < body.index(cleans[0]), (
        "`lint: clean` is printed outside the branch that checks nothing was skipped")
