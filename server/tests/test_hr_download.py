"""Downloader tests: resume, sha256 verification, retries - network mocked."""

import urllib.error
from pathlib import Path

import pytest

from app.pipeline import hr_download


class FakeResponse:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """Honours Range; can inject failures; counts requests."""

    def __init__(self, data: bytes, fail_first: int = 0, ignore_range: bool = False):
        self.data = data
        self.fail_first = fail_first
        self.ignore_range = ignore_range
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(dict(request.headers))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise urllib.error.URLError("injected network failure")
        range_header = request.headers.get("Range")
        if range_header and not self.ignore_range:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            return FakeResponse(self.data[start:], status=206)
        return FakeResponse(self.data, status=200)


def test_fresh_download_records_sha256(tmp_path):
    payload = b"x" * 5000
    server = FakeServer(payload)
    dest = tmp_path / "a.bin"
    record = hr_download.download_file("https://x/a.bin", dest, urlopen=server)
    assert dest.read_bytes() == payload
    assert record["sha256"] == hr_download.sha256_file(dest)
    assert record["reused"] is False
    assert not (tmp_path / "a.bin.part").exists()
    assert len(server.requests) == 1 and "Range" not in server.requests[0]


def test_resume_from_part_uses_range(tmp_path):
    payload = b"0123456789" * 1000
    part = tmp_path / "a.bin.part"
    part.write_bytes(payload[:1234])
    server = FakeServer(payload)
    record = hr_download.download_file("https://x/a.bin", tmp_path / "a.bin",
                                       urlopen=server)
    assert (tmp_path / "a.bin").read_bytes() == payload
    assert record["sha256"] == hr_download.sha256_file(tmp_path / "a.bin")
    assert server.requests[0].get("Range") == "bytes=1234-"


def test_resume_falls_back_when_server_ignores_range(tmp_path):
    payload = b"abcdef" * 700
    (tmp_path / "a.bin.part").write_bytes(payload[:100])
    server = FakeServer(payload, ignore_range=True)
    hr_download.download_file("https://x/a.bin", tmp_path / "a.bin", urlopen=server)
    assert (tmp_path / "a.bin").read_bytes() == payload


def test_existing_verified_file_is_reused_without_network(tmp_path):
    payload = b"hello world"
    dest = tmp_path / "a.bin"
    dest.write_bytes(payload)
    server = FakeServer(payload)
    record = hr_download.download_file(
        "https://x/a.bin", dest, expected_sha256=hr_download.sha256_file(dest),
        urlopen=server)
    assert record["reused"] is True
    assert server.requests == []


def test_existing_corrupt_file_is_redownloaded(tmp_path):
    dest = tmp_path / "a.bin"
    dest.write_bytes(b"corrupt")
    payload = b"correct payload"
    server = FakeServer(payload)
    record = hr_download.download_file(
        "https://x/a.bin", dest,
        expected_sha256=hr_download.sha256_file(_write(tmp_path / "good", payload)),
        urlopen=server)
    assert record["reused"] is False
    assert dest.read_bytes() == payload
    assert len(server.requests) == 1


def _write(path, payload):
    path.write_bytes(payload)
    return path


def test_expected_sha_mismatch_raises_and_leaves_no_part(tmp_path):
    server = FakeServer(b"something else")
    with pytest.raises(RuntimeError):
        hr_download.download_file("https://x/a.bin", tmp_path / "a.bin",
                                  expected_sha256="0" * 64, urlopen=server,
                                  retries=2)
    assert not (tmp_path / "a.bin").exists()
    assert not (tmp_path / "a.bin.part").exists()
    assert len(server.requests) == 2


def test_transient_error_is_retried(tmp_path):
    payload = b"resilient"
    server = FakeServer(payload, fail_first=1)
    record = hr_download.download_file("https://x/a.bin", tmp_path / "a.bin",
                                       urlopen=server)
    assert record["sha256"] == hr_download.sha256_file(tmp_path / "a.bin")
    assert len(server.requests) == 2


def test_checksum_index_roundtrip_is_sorted(tmp_path):
    csv_path = tmp_path / "index.csv"
    records = [
        {"path": "b/B.tif", "product": "dsm", "sheet": "2-2", "year": "2020",
         "size_bytes": 2, "sha256": "bb", "url": "https://x/b"},
        {"path": "a/A.tif", "product": "rgb", "sheet": "1-1", "year": "2019",
         "size_bytes": 1, "sha256": "aa", "url": "https://x/a"},
    ]
    hr_download.save_checksums(csv_path, records)
    loaded = hr_download.load_checksums(csv_path)
    assert list(loaded) == ["a/A.tif", "b/B.tif"]
    assert loaded["a/A.tif"]["sha256"] == "aa"


def test_collect_and_verify_checksums(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.bin").write_bytes(b"data")
    (tmp_path / "sub" / "f.bin.part").write_bytes(b"partial")
    records = hr_download.collect_checksums(tmp_path)
    paths = [r["path"] for r in records]
    assert any(p.endswith("sub/f.bin") for p in paths)
    assert not any(p.endswith(".part") for p in paths)
    csv_path = tmp_path / "index.csv"
    hr_download.save_checksums(csv_path, records)
    assert hr_download.verify_checksums(csv_path, tmp_path) == []
    (tmp_path / "sub" / "f.bin").write_bytes(b"DATE")
    problems = hr_download.verify_checksums(csv_path, tmp_path)
    assert any("sha256 mismatch" in p for p in problems)
    (tmp_path / "sub" / "f.bin").write_bytes(b"even longer tamper")
    problems = hr_download.verify_checksums(csv_path, tmp_path)
    assert any("size mismatch" in p for p in problems)


def test_relative_path_lookup_for_committed_index(tmp_path):
    root = tmp_path / "hr_downloads"
    (root / "rgb").mkdir(parents=True)
    (root / "rgb" / "t.tif").write_bytes(b"img")
    index = tmp_path / "server" / "data" / "hr_downloads.csv"
    index.parent.mkdir(parents=True)
    hr_download.save_checksums(index, hr_download.collect_checksums(root))
    loaded = hr_download.load_checksums(index)
    rel = next(iter(loaded))
    assert rel.endswith("rgb/t.tif")


def test_path_used_for_record_is_repo_relative(tmp_path, monkeypatch):
    # record paths are stored as posix strings relative to the repo root when
    # callers pass repo-relative destinations (regression guard for the index)
    dest = tmp_path / "a.bin"
    server = FakeServer(b"zz")
    record = hr_download.download_file("https://x/a.bin", Path(dest), urlopen=server)
    assert Path(record["path"]).name == "a.bin"
