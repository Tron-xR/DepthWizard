"""Resumable downloads with per-file sha256 for the HR dataset build.

Files stream into ``<dest>.part`` and are renamed on completion; resume is an
HTTP Range request from the current ``.part`` size. Every finished file's
sha256 (and byte count) goes into a CSV index that is committed to the repo,
so a fresh clone can verify the download cache without refetching.

No checksums are published on the swisstopo/Copernicus endpoints, so the
index is first-write-wins: values are recorded here the first time a file is
fetched and verified against it on every later run.
"""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from pathlib import Path

CHUNK = 1 << 20
USER_AGENT = "DepthWizard-hr-dataset/1.0 (+https://github.com/Tron-xR/DepthWizard)"
_DEFAULT_URLOPEN = urllib.request.urlopen


def sha256_file(path, chunk: int = CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checksums(csv_path) -> dict:
    """Read the checksum index into {relative_path: {size, sha256, url}}."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return {row["path"]: row for row in csv.DictReader(fh)}


def save_checksums(csv_path, records) -> None:
    """Write the index sorted by path (stable diffs for a committed file)."""
    fields = ("path", "product", "sheet", "year", "size_bytes", "sha256", "url")
    rows = [{k: row.get(k, "") for k in fields}
            for row in sorted(records, key=lambda r: r["path"])]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_checksums(root, *, product_of=None) -> list:
    """Hash every finished file under root (``.part`` files are ignored)."""
    records = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.name.endswith(".part"):
            continue
        record = {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "url": "",
        }
        if product_of is not None:
            record.update(product_of(path) or {})
        records.append(record)
    return records


def verify_checksums(csv_path, root, *, prefix_strip=("data/",)) -> list:
    """Return human-readable problems for the index vs files under root.

    Index keys may be absolute (fresh ``collect_checksums`` output) or repo-
    relative with a leading downloads-root component (the committed index).
    """
    problems = []
    index = load_checksums(csv_path)
    root = Path(root)
    for rel, row in index.items():
        stripped = rel
        if stripped.startswith(root.name + "/"):
            stripped = stripped[len(root.name) + 1:]
        else:
            for prefix in prefix_strip:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
        candidate = root / stripped
        if not candidate.exists():
            problems.append(f"missing: {rel}")
            continue
        size = candidate.stat().st_size
        if str(size) != str(row["size_bytes"]):
            problems.append(f"size mismatch: {rel} ({size} != {row['size_bytes']})")
        elif sha256_file(candidate) != row["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    return problems


class ChecksumMismatch(RuntimeError):
    pass


def download_file(url, dest, *, expected_sha256=None, urlopen=None,
                  retries: int = 3, log=None) -> dict:
    """Fetch url into dest, resuming a prior ``.part`` when possible.

    Returns {"path", "url", "size_bytes", "sha256", "reused"}. Raises
    RuntimeError after `retries` attempts; the ``.part`` file is left in a
    resumable state for network errors and removed on checksum mismatch
    (restarting from byte zero is the only safe recovery there).
    """
    fetch = urlopen or _DEFAULT_URLOPEN
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.exists():
        digest = sha256_file(dest)
        if expected_sha256 is None or digest == expected_sha256:
            return _record(url, dest, digest, reused=True)
        if log:
            log(f"[redownload] {dest.name}: sha256 mismatch with index")
        dest.unlink()

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            _fetch_into_part(url, part, fetch)
            digest = sha256_file(part)
            if expected_sha256 is not None and digest != expected_sha256:
                part.unlink()
                raise ChecksumMismatch(
                    f"{dest.name}: sha256 {digest} != expected {expected_sha256}")
            os.replace(part, dest)
            return _record(url, dest, digest, reused=False)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last_error = exc
            if log and attempt < retries:
                log(f"[retry {attempt}/{retries}] {dest.name}: {exc}")
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


def _record(url, dest, digest, *, reused: bool) -> dict:
    return {"path": dest.as_posix(), "url": url,
            "size_bytes": dest.stat().st_size, "sha256": digest, "reused": reused}


def _fetch_into_part(url, part: Path, fetch) -> None:
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with fetch(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if offset and status == 200:
            offset = 0  # server ignored the Range header: restart cleanly
        mode = "ab" if offset else "wb"
        with open(part, mode) as fh:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
    if part.stat().st_size == 0:
        part.unlink()
        raise RuntimeError(f"empty response: {url}")
