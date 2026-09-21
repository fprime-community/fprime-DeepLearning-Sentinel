"""(!) `cli.py` said "no R2 client, no credential read" and it was true of the
PACKAGE and false of the CLOSURE.

`tests/test_toolkit.py`'s `test_t7_the_toolkit_reaches_no_bucket` reads the twelve
files of `src/sentinel_toolkit/` and asserts none of them imports `boto3`,
`botocore`, `r2` or `sentinel_data`. That was true, and it gave false comfort,
because it stopped at the package boundary:

    sentinel_toolkit/selftest.py  ->  sentinel_eval.read
    sentinel_eval/read.py         ->  sentinel_data.r2      (module scope)
    sentinel_data/r2.py           ->  boto3, botocore       (module scope)

So running the offline selftest pulled a Cloudflare R2 client into the process.
It never constructed one, never read a credential and never opened a socket --
but the capability was in the import graph, and a textual check of one directory
could not see it. Found 2026-09-21 while deciding what may go on `master`.

**D80 cut the two edges** -- `read.py`'s import moved into `R2Source.get`, the one
method that needs the real transport, and `catalog.py` now spells `MANIFEST_KEY`
itself instead of importing `sentinel_data.config` for one string.

This file is the guard, and it checks the closure the only way a closure can
honestly be checked: **it makes the modules unimportable and then imports and
RUNS the toolkit.** A grep cannot do that. A subprocess is used so the blocked
`sys.modules` never leaks into the rest of the suite.

The positive control matters as much as the checks: if the blocker silently
stopped blocking, every other test here would pass while proving nothing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Everything the toolkit's closure must not need in order to import and run.
#: `dotenv` and `requests` are here because they are how a credential read and a
#: network call would arrive if one were ever added.
BLOCKED = ("sentinel_data", "boto3", "botocore", "dotenv", "requests")

BLOCKER = f'''
import sys
BLOCKED = {BLOCKED!r}

class _Blocked(Exception):
    pass

class _Finder:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ImportError(
                f"{{name}} is blocked by tests/test_toolkit_closure_is_offline.py: "
                "the ground toolkit's closure must import and run without it")
        return None

sys.meta_path.insert(0, _Finder())
sys.path.insert(0, {str(ROOT / "src")!r})
'''


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCKER + body],
                          cwd=ROOT, capture_output=True, text=True)


def test_the_blocker_actually_blocks() -> None:
    """The positive control. Without it every check below is vacuous."""
    proc = _run("import boto3\nprint('IMPORTED')\n")
    assert proc.returncode != 0, (
        "boto3 imported with the blocker installed, so the blocker is not "
        f"blocking and nothing below this line proves anything:\n{proc.stdout}")
    assert "is blocked by" in proc.stderr, proc.stderr

    proc = _run("import numpy\nprint('OK')\n")
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"the blocker blocks more than it should -- numpy must still import:\n"
        f"{proc.stdout}{proc.stderr}")


def test_the_whole_toolkit_closure_imports_with_no_cloud_client_available() -> None:
    """`fit`, `verify` and `selftest`, all three entry points, by import."""
    proc = _run(
        "import sentinel_toolkit.cli\n"
        "import sentinel_toolkit.fit\n"
        "import sentinel_toolkit.selftest\n"
        "import sentinel_eval.read, sentinel_eval.catalog, sentinel_eval.bundle\n"
        "import sentinel_export\n"
        "print('IMPORTED')\n")
    assert proc.returncode == 0 and "IMPORTED" in proc.stdout, (
        "the toolkit's closure cannot be imported without a cloud client "
        f"present:\n{proc.stdout}\n{proc.stderr}")


def test_no_blocked_module_reaches_sys_modules_even_indirectly() -> None:
    """Import alone is not the whole claim: nothing may be pulled in lazily
    during the import of anything else either."""
    proc = _run(
        "import sentinel_toolkit.cli, sentinel_toolkit.selftest\n"
        # (!) The harness modules too. Importing only the toolkit package would
        # pass on the PRE-D80 chain, because selftest's `sentinel_eval` imports
        # are function-local -- the leak happened when they ran, not when the
        # package loaded. A check that cannot see the defect it was written for
        # is the thing this whole file exists to object to.
        "import sentinel_eval.read, sentinel_eval.catalog, sentinel_eval.bundle\n"
        "import sys\n"
        f"leaked = sorted(m for m in sys.modules if m.split('.')[0] in {BLOCKED!r})\n"
        "print('LEAKED', leaked)\n")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LEAKED []" in proc.stdout, (
        f"a blocked module reached sys.modules anyway:\n{proc.stdout}")


def test_the_selftest_actually_runs_offline_end_to_end() -> None:
    """(!) The check that a grep cannot make. Roughly five seconds."""
    proc = _run(
        "from sentinel_toolkit.selftest import run\n"
        "import io, contextlib\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    code = run()\n"
        "out = buf.getvalue()\n"
        "print('EXIT', code)\n"
        "print([l for l in out.splitlines() if 'checks passed' in l])\n")
    assert proc.returncode == 0, (
        f"the selftest did not run with the cloud client blocked:\n"
        f"{proc.stdout}\n{proc.stderr}")
    assert "EXIT 0" in proc.stdout, f"the selftest did not pass:\n{proc.stdout}"
    assert "8/8 checks passed" in proc.stdout, (
        f"the selftest ran offline but no longer reports 8/8:\n{proc.stdout}")


def test_the_duplicated_manifest_key_has_not_drifted() -> None:
    """D80 spells `MANIFEST_KEY` in `sentinel_eval/catalog.py` rather than
    importing it, to cut the dependency. The ingest side still owns the value,
    so the two spellings are asserted equal here -- a duplicated constant with a
    guard, which is the trade D80 records.

    Runs WITHOUT the blocker: on `dev` both sides exist, and that is the point.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from sentinel_data import config as C
    from sentinel_eval.catalog import Catalog

    assert Catalog.MANIFEST_KEY == C.MANIFEST_KEY, (
        f"sentinel_eval/catalog.py spells the manifest key "
        f"{Catalog.MANIFEST_KEY!r} and sentinel_data/config.py spells it "
        f"{C.MANIFEST_KEY!r}. One of them moved; D80 requires they agree.")


def test_the_toolkit_requirements_are_pinned_the_same_and_carry_no_cloud_client() -> None:
    """`requirements-toolkit.txt` is what `master` ships. D80.

    Two ways it could go wrong and both are checked: a pin drifting away from
    the repository-wide list, and a cloud dependency creeping back in. The
    second is the one that matters -- a dependency list that installs boto3
    tells a reader the opposite of what D80 established.
    """
    def pins(name: str) -> dict[str, str]:
        out = {}
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pkg, _, ver = line.partition("==")
                out[pkg.strip().lower()] = ver.strip()
        return out

    toolkit, whole = pins("requirements-toolkit.txt"), pins("requirements.txt")

    assert set(toolkit) == {"numpy", "pyarrow", "torch"}, (
        f"the toolkit closure needs numpy, pyarrow and torch and nothing else "
        f"(checked by importing it with everything else blocked); the list says "
        f"{sorted(toolkit)}")
    for banned in ("boto3", "botocore", "requests", "python-dotenv", "s3fs"):
        assert banned not in toolkit, (
            f"requirements-toolkit.txt installs {banned}. The toolkit reaches no "
            "bucket and reads no credential -- D80 -- and this list is on `master`.")
    drifted = {p: (v, whole.get(p)) for p, v in toolkit.items() if whole.get(p) != v}
    assert not drifted, (
        f"pins differ between requirements-toolkit.txt and requirements.txt: "
        f"{drifted}. They are the same software; pin them once.")
