"""Reading the ESA-ADB mission archives.

Three facts drive this module:

1. The per-channel files are inner .zip archives STORED (method 0) inside the
   outer archive. They are read in place -- no unpacking step, and no pickle is
   ever written to disk.
2. channels.csv alone is compressed with Deflate64 (method 9), which Python's
   stdlib zipfile refuses. Deflate64 only diverges from Deflate for large
   windows and long matches, so for a file this small the stream is plain-
   inflate compatible: raw zlib decodes it and the CRC proves it. An external
   `unzip` is kept as a fallback for the case where it ever is not.
   (bsdtar/libarchive does NOT support Deflate64 -- verified, it errors.)
3. Channel values are not always numeric. Some channels are categorical and
   unpickle as string dtype, so value dtype is preserved, never coerced.
"""
from __future__ import annotations

import binascii
import io
import re
import struct
import subprocess
import zlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFLATE64 = 9
_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_SIG = 0x04034B50


class Deflate64Error(RuntimeError):
    pass


def _natural_key(name: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _synthetic_zip(name: str, info: zipfile.ZipInfo, raw: bytes) -> bytes:
    """Wrap one raw, still-compressed member in a minimal single-entry ZIP."""
    nb = name.encode()
    dosdate = (1 << 5) | 1
    local = _LOCAL_HEADER.pack(
        _LOCAL_SIG, 45, 0, info.compress_type, 0, dosdate,
        info.CRC, info.compress_size, info.file_size, len(nb), 0,
    ) + nb
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII", 0x02014B50, 45, 45, 0, info.compress_type, 0, dosdate,
        info.CRC, info.compress_size, info.file_size, len(nb), 0, 0, 0, 0, 0, 0,
    ) + nb
    eocd = struct.pack(
        "<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local) + len(raw), 0
    )
    return local + raw + central + eocd


@dataclass
class MissionArchive:
    """One ESA-MissionN archive, over a local path or an HTTP range file."""

    zf: zipfile.ZipFile
    mission: str
    archive_path: Path | None = None
    scratch: Path | None = None

    @property
    def root(self) -> str:
        return self.zf.namelist()[0].split("/", 1)[0]

    # -- membership --------------------------------------------------------
    def _members(self, folder: str) -> list[str]:
        prefix = f"{self.root}/{folder}/"
        names = [n for n in self.zf.namelist() if n.startswith(prefix) and n.endswith(".zip")]
        return sorted(names, key=_natural_key)

    def channel_members(self) -> list[str]:
        return self._members("channels")

    def telecommand_members(self) -> list[str]:
        return self._members("telecommands")

    @staticmethod
    def id_of(member: str) -> str:
        return Path(member).name[: -len(".zip")]

    def has_member(self, name: str) -> bool:
        try:
            self.zf.getinfo(f"{self.root}/{name}")
            return True
        except KeyError:
            return False

    # -- raw member access -------------------------------------------------
    def read_bytes(self, name: str) -> bytes:
        full = name if name.startswith(f"{self.root}/") else f"{self.root}/{name}"
        info = self.zf.getinfo(full)
        if info.compress_type != DEFLATE64:
            return self.zf.read(full)
        return self._read_deflate64(full, info)

    def _raw_member_bytes(self, info: zipfile.ZipInfo) -> bytes:
        fh = self.zf.fp
        fh.seek(info.header_offset)
        fields = _LOCAL_HEADER.unpack(fh.read(_LOCAL_HEADER.size))
        if fields[0] != _LOCAL_SIG:
            raise Deflate64Error(f"bad local header at offset {info.header_offset}")
        name_len, extra_len = fields[9], fields[10]
        fh.seek(info.header_offset + _LOCAL_HEADER.size + name_len + extra_len)
        return fh.read(info.compress_size)

    def _read_deflate64(self, full: str, info: zipfile.ZipInfo) -> bytes:
        raw = self._raw_member_bytes(info)

        # Preferred path: raw inflate, proven correct by the stored CRC.
        try:
            out = zlib.decompress(raw, -15)
            if len(out) == info.file_size and (binascii.crc32(out) & 0xFFFFFFFF) == info.CRC:
                return out
        except zlib.error:
            pass

        # Fallback: a real Deflate64 decoder, via a synthetic one-entry zip.
        scratch = self.scratch or Path(".")
        scratch.mkdir(parents=True, exist_ok=True)
        tmp = scratch / "_deflate64.zip"
        tmp.write_bytes(_synthetic_zip(Path(full).name, info, raw))
        try:
            proc = subprocess.run(
                ["unzip", "-p", str(tmp), Path(full).name], capture_output=True, check=False
            )
            out = proc.stdout
            if proc.returncode == 0 and (binascii.crc32(out) & 0xFFFFFFFF) == info.CRC:
                return out
            raise Deflate64Error(
                f"could not decode Deflate64 member {full}: "
                f"rc={proc.returncode} {proc.stderr.decode()[:300]}"
            )
        finally:
            tmp.unlink(missing_ok=True)

    # -- typed readers -----------------------------------------------------
    def read_csv(self, name: str, **kw) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read_bytes(name)), **kw)

    def read_series(self, member: str) -> pd.DataFrame:
        """Load one channel or telecommand as a tidy two-column frame.

        The member is an inner zip holding a single zip-compressed pickle of a
        pandas DataFrame: a tz-naive DatetimeIndex and one column named after
        the channel id. Value dtype is preserved exactly.
        """
        inner_bytes = self.zf.read(member)
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            names = inner.namelist()
            if len(names) != 1:
                raise RuntimeError(f"{member}: expected 1 inner member, found {names}")
            payload = inner.read(names[0])
        del inner_bytes

        obj = pd.read_pickle(io.BytesIO(payload))
        del payload

        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        if obj.shape[1] != 1:
            raise RuntimeError(f"{member}: expected 1 column, found {list(obj.columns)}")

        return pd.DataFrame(
            {"timestamp": pd.to_datetime(obj.index), "value": obj.iloc[:, 0].to_numpy()}
        )


def open_local(path: Path, mission: str, scratch: Path | None = None) -> MissionArchive:
    return MissionArchive(zipfile.ZipFile(path), mission, archive_path=path, scratch=scratch)


def open_fileobj(fileobj, mission: str, scratch: Path | None = None) -> MissionArchive:
    return MissionArchive(zipfile.ZipFile(fileobj), mission, archive_path=None, scratch=scratch)
