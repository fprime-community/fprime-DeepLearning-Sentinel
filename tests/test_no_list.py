"""The source-level ban the brief believed already existed."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_no_list  # noqa: E402


def test_the_repository_is_clean():
    assert check_no_list.main() == 0


def test_the_check_actually_catches_things(tmp_path, monkeypatch):
    probe = ROOT / "src" / "sentinel_eval" / "_probe_check_no_list.py"
    probe.write_text(
        "import boto3\n"
        "import s3fs\n"
        "def bad(client, bucket):\n"
        "    client.list_objects_v2(Bucket=bucket)\n"
        "    return list(Path('x').glob('*.parquet'))\n"
        "SNAPSHOT = 'docs/manifest.snapshot.json'\n"
    )
    try:
        problems = check_no_list.check_file(probe)
        rules = {p.split(": ", 1)[1].split("\n")[0] for p in problems}
        assert len(problems) == 4
        assert any("LIST-class" in r for r in rules)
        assert any("s3fs/fsspec" in r for r in rules)
        assert any(".glob()" in r for r in rules)
        assert any("snapshot" in r for r in rules)
        assert check_no_list.main() == 1
    finally:
        probe.unlink()


def test_local_globbing_outside_r2_modules_is_allowed():
    """`transcode.py` globs a scratch directory; that is not a bucket LIST."""
    assert check_no_list.check_file(ROOT / "src" / "sentinel_data" / "transcode.py") == []


def test_the_snapshot_writer_is_exempted_visibly():
    assert ("src/sentinel_data/docs_gen.py", "snapshot") in check_no_list.ALLOWED
