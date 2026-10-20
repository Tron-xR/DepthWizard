"""Unit tests for the SQLite metadata store (06-database.md)."""
import sqlite3

from app import db
from app.schemas import EvaluationResponse


def _result():
    uid = db.create_upload(
        original_filename="p.png", file_path="/c/p.png", file_type="png",
        is_georeferenced=False,
    )["id"]
    job = db.create_job(uid, "rdsm")
    return db.create_result(
        job_id=job["id"], heightmap_path="/x/h.png", texture_path="/x/t.png",
        min_elev=0.0, max_elev=1.0, world_width=1000.0, world_depth=1000.0,
    )


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

    db.create_evaluation(
        result_id=res["id"],
        reference_source="SRTM_30m",
        rmse=4.2,
        mae=3.1,
        correlation=0.94,
    )
    assert db.get_evaluations_for_result(res["id"])[0]["rmse"] == 4.2


def test_evaluation_diagnostics_round_trip_and_nonfinite_are_null():
    res = _result()
    db.create_evaluation(
        result_id=res["id"],
        reference_source="local",
        rmse=4.2,
        mae=3.1,
        correlation=0.94,
        raw_correlation=-0.94,
        scale_sign="-",
        polarity_inverted=True,
        calibration_warning="negative_scale",
        estimate_status="low_confidence",
        bias=-85.3164,
    )
    ev = db.get_evaluations_for_result(res["id"])[0]
    assert ev["raw_correlation"] == -0.94
    assert ev["scale_sign"] == "-"
    assert ev["polarity_inverted"] is True
    assert ev["calibration_warning"] == "negative_scale"
    assert ev["estimate_status"] == "low_confidence"
    assert ev["bias"] == -85.3164

    db.create_evaluation(
        result_id=res["id"],
        reference_source="local",
        raw_correlation=float("nan"),
        bias=float("inf"),
    )
    ev = db.get_evaluations_for_result(res["id"])[-1]
    assert ev["raw_correlation"] is None
    assert ev["bias"] is None
    assert ev["scale_sign"] is None
    assert ev["polarity_inverted"] is None


def test_init_db_migrates_old_evaluation_rows_in_place(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE evaluation_results (
               id TEXT PRIMARY KEY,
               result_id TEXT NOT NULL,
               reference_source TEXT NOT NULL,
               landscape_type TEXT,
               rmse REAL,
               mae REAL,
               correlation REAL
           )"""
    )
    con.execute(
        """INSERT INTO evaluation_results
           (id, result_id, reference_source, landscape_type,
            rmse, mae, correlation)
           VALUES ('e1', 'r1', 'SRTM_30m', NULL, 4.2, 3.1, 0.94)"""
    )
    con.commit()
    con.close()

    db.init_db(path)
    db.init_db(path)

    with db.connect(path) as con:
        cols = {
            row["name"]
            for row in con.execute("PRAGMA table_info(evaluation_results)")
        }
        old = dict(con.execute("SELECT * FROM evaluation_results").fetchone())

    assert {
        "raw_correlation", "scale_sign", "polarity_inverted",
        "calibration_warning", "estimate_status", "bias",
    } <= cols
    assert old["rmse"] == 4.2
    assert old["mae"] == 3.1
    assert old["correlation"] == 0.94
    assert old["raw_correlation"] is None
    assert old["scale_sign"] is None
    assert old["polarity_inverted"] is None
    assert old["calibration_warning"] is None
    assert old["estimate_status"] is None
    assert old["bias"] is None


def test_archived_hilly10_diagnostics_survive_schema_and_db():
    resp = EvaluationResponse(
        mae=173.2479378399383,
        rmse=259.551960848074,
        correlation=0.6969701685813597,
        raw_correlation_signed=-0.6969701685813604,
        scale_sign="-",
        polarity_inverted=True,
        calibration_warning="negative_scale",
        estimate_status="low_confidence",
        bias=-85.3164,
        valid_pixels=103680,
        total_pixels=103680,
        held_out_pixel_count=103680,
        calibrated=True,
        evaluated_units="calibrated absolute elevation",
    )
    assert resp.raw_correlation_signed < 0 < resp.correlation

    res = _result()
    db.create_evaluation(
        result_id=res["id"],
        reference_source="archived",
        rmse=resp.rmse,
        mae=resp.mae,
        correlation=resp.correlation,
        raw_correlation=resp.raw_correlation_signed,
        scale_sign=resp.scale_sign,
        polarity_inverted=resp.polarity_inverted,
        calibration_warning=resp.calibration_warning,
        estimate_status=resp.estimate_status,
        bias=resp.bias,
    )
    ev = db.get_evaluations_for_result(res["id"])[-1]
    assert ev["correlation"] == resp.correlation
    assert ev["raw_correlation"] == resp.raw_correlation_signed
    assert ev["scale_sign"] == "-"
    assert ev["polarity_inverted"] is True
    assert ev["calibration_warning"] == "negative_scale"
    assert ev["estimate_status"] == "low_confidence"
    assert ev["bias"] == -85.3164


def test_upload_not_found():
    assert db.get_upload("nope") is None
