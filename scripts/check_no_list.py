#!/usr/bin/env python3
"""Source-level ban on LIST-class access to R2. Fails the build, not the bill.

`sentinel_data.r2.OpCounter` already refuses a LIST at runtime, but a runtime
guard only fires on a code path someone actually executed. A single
`fsspec`-backed glob in a rarely-taken branch is a paginated LIST that nobody
sees until the bill arrives. This check reads the source instead.

Three rules, each scoped so that legitimate code is not collateral damage:

1. **No LIST method calls.** Matched with a leading dot, so `r2.py`'s own
   classification tables -- which must name the operations in order to ban them
   -- are not flagged.
2. **No `s3fs` / `fsspec`.** Both silently expand glob patterns into LIST.
   Matched as imports and attribute access, so prose in `docs_gen.py` that warns
   about them is not flagged.
3. **No `.glob()` in a module that reaches R2.** `transcode.py` globs a local
   scratch directory, which is fine; the rule applies only to files that import
   boto3 or the R2 client, so it separates the two without an allowlist.

Plus: no code may read `docs/manifest.snapshot.json`. It is a reference copy for
humans; R2's `_manifest/manifest.json` is the only authority.

Exemptions live in :data:`ALLOWED` with a stated reason, and are printed on every
clean run. An exemption nobody can see is an exemption nobody re-examines.

Exit code 0 clean, 1 on any violation. `tests/test_no_list.py` calls `main()`,
so the suite fails too.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "scripts")
SELF = Path(__file__).name

# A module reaches R2 if it can construct or use the client.
R2_MARKERS = ("boto3", "sentinel_data.r2", "from .r2", "from sentinel_data import", " r2.")

LIST_CALLS = re.compile(
    r"\.\s*(list_objects|list_objects_v2|list_buckets|list_multipart_uploads|list_parts)\s*\(",
    re.IGNORECASE,
)
FSSPEC = re.compile(r"^\s*(?:from|import)\s+(s3fs|fsspec)\b|(?<![\w`])(s3fs|fsspec)\s*\.")
GLOB_CALL = re.compile(r"\.\s*glob\s*\(")
SNAPSHOT = re.compile(r"manifest\.snapshot")

#: (path, rule) -> why. `docs_gen` is the snapshot's *writer*: it regenerates it
#: from the manifest that was just published, which is the opposite of treating
#: it as an authority. Nothing else may name it at all.
ALLOWED: dict[tuple[str, str], str] = {
    ("src/sentinel_data/docs_gen.py", "snapshot"):
        "generates the snapshot from the published manifest; it writes, never reads",
}


def touches_r2(source: str) -> bool:
    return any(marker in source for marker in R2_MARKERS)


def check_file(path: Path) -> list[str]:
    source = path.read_text()
    rel = path.relative_to(ROOT).as_posix()
    r2_module = touches_r2(source)
    problems: list[str] = []

    def flag(rule: str, lineno: int, line: str, message: str) -> None:
        if (rel, rule) in ALLOWED:
            return
        problems.append(f"{rel}:{lineno}: {message}\n    {line.strip()}")

    for lineno, line in enumerate(source.splitlines(), 1):
        if LIST_CALLS.search(line):
            flag("list", lineno, line, "LIST-class call -- resolve keys from the manifest")
        if FSSPEC.search(line):
            flag("fsspec", lineno, line, "s3fs/fsspec expands globs into LIST silently")
        if r2_module and GLOB_CALL.search(line):
            flag("glob", lineno, line, ".glob() in a module that reaches R2")
        if SNAPSHOT.search(line):
            flag("snapshot", lineno, line,
                 "code must read R2's _manifest/manifest.json, not the snapshot")
    return problems


def main(argv=None) -> int:
    problems: list[str] = []
    scanned = 0
    for folder in SCANNED:
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path.name == SELF or "__pycache__" in path.parts:
                continue
            scanned += 1
            problems.extend(check_file(path))

    if problems:
        print(f"BANNED R2 ACCESS PATTERNS -- {len(problems)} in {scanned} files\n")
        for problem in problems:
            print(f"  {problem}")
        print("\nEvery key comes from the manifest. See docs/DATA.md section 4.")
        return 1

    print(f"check_no_list: {scanned} files clean -- no LIST, no glob, no snapshot reads")
    for (path, rule), reason in sorted(ALLOWED.items()):
        print(f"  exemption  {path} [{rule}]: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
