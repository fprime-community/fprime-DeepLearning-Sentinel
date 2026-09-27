"""`DirSource`: the bucket's layout on files, for a machine that must not reach R2 (D86)."""
import hashlib

import pytest

from sentinel_eval.catalog import StoredObject
from sentinel_eval.errors import IntegrityError
from sentinel_eval.read import DirSource, fetch_object


class Counting:
    def __init__(self, blobs):
        self.blobs, self.calls = blobs, 0

    def get(self, key):
        self.calls += 1
        return self.blobs[key]


def test_reads_a_mirrored_key(tmp_path):
    (tmp_path / "a" / "b.npy").parent.mkdir(parents=True)
    (tmp_path / "a" / "b.npy").write_bytes(b"xyz")
    assert DirSource(tmp_path).get("a/b.npy") == b"xyz"


def test_a_missing_key_is_an_error_not_a_fetch(tmp_path):
    with pytest.raises(FileNotFoundError):
        DirSource(tmp_path).get("never/mirrored.npy")


def test_a_key_cannot_escape_the_mirror(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "secret").write_bytes(b"no")
    with pytest.raises(IntegrityError):
        DirSource(tmp_path / "root").get("../secret")


def test_the_fallback_fills_once_and_the_file_serves_after(tmp_path):
    inner = Counting({"k/v.bin": b"payload"})
    src = DirSource(tmp_path, fallback=inner)
    assert src.get("k/v.bin") == b"payload"
    assert src.get("k/v.bin") == b"payload"
    assert inner.calls == 1
    assert DirSource(tmp_path).get("k/v.bin") == b"payload"
    assert not list(tmp_path.rglob("*.part"))


def test_the_manifest_checksum_still_guards_a_mirrored_file(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"tampered")
    obj = StoredObject("x.bin", hashlib.sha256(b"original").hexdigest(), 8, None)
    with pytest.raises(IntegrityError):
        fetch_object(DirSource(tmp_path), obj)
