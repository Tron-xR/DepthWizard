"""SQLite metadata store for uploads, jobs, results, and evaluation results.

Schema follows 06-database.md. Single local user, so a plain sqlite3 connector
with row wrappers is sufficient (no ORM). All methods are synchronous.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    is_georeferenced INTEGER NOT NULL,
    crs TEXT,
    bounds TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    upload_id TEXT NOT NULL REFERENCES uploads(id),
    status TEXT NOT NULL,
    branch TEXT NOT NULL,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    heightmap_path TEXT NOT NULL,
    texture_path TEXT NOT NULL,
    dsm_geotiff_path TEXT,
    min_elev REAL,
    max_elev REAL,
    cell_size REAL,
    world_width REAL,
    world_depth REAL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES results(id),
    reference_source TEXT NOT NULL,
    landscape_type TEXT,
    rmse REAL,
    mae REAL,
    correlation REAL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def connect(db_path: Optional[Any] = None) -> Iterator[sqlite3.Connection]:
    """Yield a committed-on-success connection. Plain connections have autocommit
    off in python 3.10 unless isolation_level queries; we manage commits manually."""
    path = db_path if db_path is not None else config.DB_PATH
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(db_path: Optional[Any] = None) -> None:
    with connect(db_path) as con:
        con.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# uploads
# --------------------------------------------------------------------------- #
def create_upload(
    *,
    original_filename: str,
    file_path: str,
    file_type: str,
    is_georeferenced: bool,
    crs: Optional[str] = None,
    bounds: Optional[list] = None,
    db_path: Optional[Any] = None,
) -> dict:
    row = {
        "id": _new_id(),
        "original_filename": original_filename,
        "file_path": file_path,
        "file_type": file_type,
        "is_georeferenced": int(is_georeferenced),
        "crs": crs,
        "bounds": json.dumps(bounds) if bounds is not None else None,
        "created_at": _now(),
    }
    with connect(db_path) as con:
        con.execute(
            """INSERT INTO uploads
               (id, original_filename, file_path, file_type, is_georeferenced,
                crs, bounds, created_at)
               VALUES (:id, :original_filename, :file_path, :file_type,
                :is_georeferenced, :crs, :bounds, :created_at)""",
            row,
        )
    return _row_to_dict(row)


def get_upload(upload_id: str, db_path: Optional[Any] = None) -> Optional[dict]:
    with connect(db_path) as con:
        row = con.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    return _decode_row(row)


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
def create_job(upload_id: str, branch: str, db_path: Optional[Any] = None) -> dict:
    row = {
        "id": _new_id(),
        "upload_id": upload_id,
        "status": "queued",
        "branch": branch,
        "error_message": None,
        "started_at": _now(),
        "completed_at": None,
    }
    with connect(db_path) as con:
        con.execute(
            """INSERT INTO jobs
               (id, upload_id, status, branch, error_message, started_at, completed_at)
               VALUES (:id, :upload_id, :status, :branch, :error_message,
                :started_at, :completed_at)""",
            row,
        )
    return _row_to_dict(row)


def get_job(job_id: str, db_path: Optional[Any] = None) -> Optional[dict]:
    with connect(db_path) as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _decode_row(row)


def set_job_status(job_id: str, status: str, error: Optional[str] = None, db_path: Optional[Any] = None):
    with connect(db_path) as con:
        con.execute(
            """UPDATE jobs SET status = ?, error_message = ?,
               completed_at = COALESCE(completed_at, ?) WHERE id = ?""",
            (status, error, _now() if status in ("done", "failed") else None, job_id),
        )


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def create_result(
    *,
    job_id: str,
    heightmap_path: str,
    texture_path: str,
    dsm_geotiff_path: Optional[str] = None,
    min_elev: Optional[float] = None,
    max_elev: Optional[float] = None,
    cell_size: Optional[float] = None,
    world_width: Optional[float] = None,
    world_depth: Optional[float] = None,
    db_path: Optional[Any] = None,
) -> dict:
    row = {
        "id": _new_id(),
        "job_id": job_id,
        "heightmap_path": heightmap_path,
        "texture_path": texture_path,
        "dsm_geotiff_path": dsm_geotiff_path,
        "min_elev": min_elev,
        "max_elev": max_elev,
        "cell_size": cell_size,
        "world_width": world_width,
        "world_depth": world_depth,
    }
    with connect(db_path) as con:
        cur = con.execute(
            """INSERT INTO results
               (id, job_id, heightmap_path, texture_path, dsm_geotiff_path,
                min_elev, max_elev, cell_size, world_width, world_depth)
               VALUES (:id, :job_id, :heightmap_path, :texture_path,
                :dsm_geotiff_path, :min_elev, :max_elev, :cell_size,
                :world_width, :world_depth)""",
            row,
        )
    return _row_to_dict(row)


def get_result_by_job(job_id: str, db_path: Optional[Any] = None) -> Optional[dict]:
    with connect(db_path) as con:
        row = con.execute("SELECT * FROM results WHERE job_id = ?", (job_id,)).fetchone()
    return _decode_row(row)


# --------------------------------------------------------------------------- #
# evaluation_results
# --------------------------------------------------------------------------- #
def create_evaluation(
    *,
    result_id: str,
    reference_source: str,
    landscape_type: Optional[str] = None,
    rmse: Optional[float] = None,
    mae: Optional[float] = None,
    correlation: Optional[float] = None,
    db_path: Optional[Any] = None,
) -> dict:
    row = {
        "id": _new_id(),
        "result_id": result_id,
        "reference_source": reference_source,
        "landscape_type": landscape_type,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
    }
    with connect(db_path) as con:
        con.execute(
            """INSERT INTO evaluation_results
               (id, result_id, reference_source, landscape_type, rmse, mae, correlation)
               VALUES (:id, :result_id, :reference_source, :landscape_type,
                :rmse, :mae, :correlation)""",
            row,
        )
    return _row_to_dict(row)


def get_evaluations_for_result(result_id: str, db_path: Optional[Any] = None) -> list[dict]:
    with connect(db_path) as con:
        rows = con.execute(
            "SELECT * FROM evaluation_results WHERE result_id = ?", (result_id,)
        ).fetchall()
    return [_decode_row(r) for r in rows]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _decode_row(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if row is None:
        return None
    d = dict(row)
    # normalize integer booleans
    if "is_georeferenced" in d and d["is_georeferenced"] is not None:
        d["is_georeferenced"] = bool(d["is_georeferenced"])
    if d.get("bounds"):
        d["bounds"] = json.loads(d["bounds"])
    return d


def _row_to_dict(row: dict) -> dict:
    return dict(row)
