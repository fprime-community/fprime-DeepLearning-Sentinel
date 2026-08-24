"""Zenodo access: resumable full downloads, and random access over HTTP.

The random-access half exists so the Phase 0 spike can pull a single ~80 MB
channel out of a 3.7 GB archive without downloading the archive.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import time
import sys
from pathlib import Path

import requests

from .config import SourceFile

CHUNK = 8 * 1024 * 1024


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file object backed by HTTP Range requests.

    Handing this to zipfile.ZipFile lets the stdlib parse the central directory
    (including ZIP64, which these >4 GB archives require) and extract a single
    member, while only the bytes actually touched cross the network.
    """

    def __init__(self, url: str, size: int, session: requests.Session | None = None):
        self._url = url
        self._size = size
        self._pos = 0
        self._session = session or requests.Session()
        self.bytes_fetched = 0
        self.requests_made = 0

    # -- io plumbing -------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0:
            return b""
        start, end = self._pos, self._pos + n - 1
        resp = self._session.get(
            self._url, headers={"Range": f"bytes={start}-{end}"}, timeout=60, stream=True
        )
        resp.raise_for_status()
        if resp.status_code != 206:
            raise RuntimeError(
                f"server ignored Range request (status {resp.status_code}); "
                "random access is unavailable"
            )
        data = resp.content
        self._pos += len(data)
        self.requests_made += 1
        self.bytes_fetched += len(data)
        return data

    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote(src: SourceFile) -> io.BufferedReader:
    """Buffered random access to a Zenodo archive without downloading it."""
    raw = HttpRangeFile(src.url, src.size)
    return io.BufferedReader(raw, buffer_size=4 * 1024 * 1024)  # type: ignore[arg-type]


def download(src: SourceFile, dest_dir: Path, verify: bool = True, max_stalls: int = 12) -> Path:
    """Download one archive, resuming across as many attempts as it takes.

    Two Zenodo behaviours shape this:

    * Long-lived connections silently wedge -- the socket stays open and
      transfers nothing, without tripping a read timeout. --speed-limit with
      --speed-time makes curl abandon a connection that goes quiet.
    * curl's own --retry must NOT be used together with --continue-at, because
      curl fixes the resume offset once at startup and every retry rewrites from
      that same point. That livelocks: the file oscillates around one offset and
      never advances. So retries are driven from here instead, re-invoking curl
      so it recomputes the offset from the bytes actually on disk each time.

    Zenodo publishes MD5 only -- there is no SHA-256 in the record -- so MD5 is
    what source verification can use.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.key
    if dest.exists() and dest.stat().st_size > src.size:
        dest.unlink()

    stalls = 0
    while not dest.exists() or dest.stat().st_size < src.size:
        before = dest.stat().st_size if dest.exists() else 0
        print(
            f"  {src.key}: resuming at {before/1e9:.2f} / {src.size/1e9:.2f} GB",
            file=sys.stderr, flush=True,
        )
        subprocess.run([
            "curl", "--location", "--continue-at", "-",
            "--speed-limit", "51200", "--speed-time", "30",
            "--connect-timeout", "20", "--progress-bar",
            "--output", str(dest), src.url,
        ])
        after = dest.stat().st_size if dest.exists() else 0
        if after >= src.size:
            break
        if after <= before:
            stalls += 1
            if stalls >= max_stalls:
                raise RuntimeError(
                    f"{src.key}: no progress past {after} bytes after {stalls} attempts"
                )
            time.sleep(min(2 ** stalls, 60))
        else:
            stalls = 0

    actual = dest.stat().st_size
    if actual != src.size:
        raise RuntimeError(f"{src.key}: size {actual} != expected {src.size}")

    if verify:
        digest = md5_file(dest)
        if digest != src.md5:
            raise RuntimeError(f"{src.key}: MD5 {digest} != published {src.md5}")
        print(f"  {src.key}: MD5 verified {digest}", file=sys.stderr, flush=True)
    return dest


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()
