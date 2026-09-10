"""Unit tests for the SQLite metadata store (06-database.md)."""
from app import db


def test_upload_crud():
    uid = db.create_upload(
        original_filename="a.tif",
        file_path="/cache/a.tif",
        file_type="tiff",
        is_georeferenced=True,
        crs="EPSG:32633",
        bounds=[500000, 4649000, 501000, 4650000],
    )["id"]
    u = db.get_upload(uid)
    assert u["is_georeferenced"] is True
    assert u["crs"] == "EPSG:32633"
    assert u["bounds"] == [500000, 4649000, 501000, 4650000]


def test_job_result_evaluation_chain():
    uid = db.create_upload(
        original_filename="p.png", file_path="/c/p.png", file_type="png",
        is_georeferenced=False,
    )["id"]
    job = db.create_job(uid, "rdsm")
    assert job["status"] == "queued"
    db.set_job_status(job["id"], "running")
    assert db.get_job(job["id"])["status"] == "running"
    db.set_job_status(job["id"], "done")
    assert db.get_job(job["id"])["status"] == "done"

    res = db.create_result(
        job_id=job["id"], heightmap_path="/x/h.png", texture_path="/x/t.png",
        min_elev=0.0, max_elev=1.0, world_width=1000.0, world_depth=1000.0,
    )
    assert db.get_result_by_job(job["id"])["id"] == res["id"]

    ev = db.create_evaluation(
        result_id=res["id"], reference_source="SRTM_30m", rmse=4.2, mae=3.1, correlation=0.94,
    )
    assert db.get_evaluations_for_result(res["id"])[0]["rmse"] == 4.2


def test_upload_not_found():
    assert db.get_upload("nope") is None
