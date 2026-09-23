"""Integration test for the georeferenced (absolute DSM) branch.

Uses the DEPTHWIZARD_DEM_FILE override so no network is required. We build a
synthetic GeoTIFF whose elevation is a known affine of the (mocked) relative
depth, and verify the calibration recovers near-real absolute elevation and
produces a georeferenced DSM GeoTIFF + validation metrics.
"""
import time
from pathlib import Path

import numpy as np
import pytest

from app import config, db, jobs
from app.pipeline import calibration as calib, depth
from app.pipeline.validate import compute_metrics
from tests.conftest import write_geotiff


def _fake_rdsm(rgb, **_kw):
    """Mimic the real depth model: return a 2D (H,W) relative surface."""
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    return np.asarray(xx / max(w - 1, 1) * 0.5 + yy / max(h - 1, 1) * 0.25,
                      dtype="float32")


@pytest.fixture
def geo_environment(tmp_path, monkeypatch):
    """Set up a local reference DEM override + georeferenced upload."""
    # local reference DEM (the "true" elevation) as a GeoTIFF
    yy, xx = np.mgrid[0:40, 0:40].astype("float32")
    true_elev = 150.0 + 0.7 * xx + 0.3 * yy  # gentle slope in meters
    ref_path = tmp_path / "ref_dem.tif"
    write_geotiff(ref_path, true_elev, minx=500000.0, maxy=4650000.0, cell=30.0)
    monkeypatch.setenv("DEPTHWIZARD_DEM_FILE", str(ref_path))

    # create a georeferenced upload matching those bounds
    rgb = true_elev.copy()  # use elevation as a stand-in texture (RGB is irrelevant here)
    upload = db.create_upload(
        original_filename="geo.tif",
        file_path=str(ref_path),  # load_raster works on this geotiff
        file_type="tiff",
        is_georeferenced=True,
        crs="EPSG:32633",
        bounds=[500000.0, 4648800.0, 501200.0, 4650000.0],
    )
    return upload, true_elev


def test_absolute_branch_produces_calibrated_dsm(geo_environment, monkeypatch):
    upload, true_elev = geo_environment
    monkeypatch.setattr(depth, "infer_relative_dsm", _fake_rdsm)

    job = db.create_job(upload["id"], "absolute_dsm")
    jobs._run_job(job["id"])

    done = db.get_job(job["id"])
    assert done["status"] == "done", done.get("error_message")

    res = db.get_result_by_job(job["id"])
    assert res["dsm_geotiff_path"] is not None
    assert res["cell_size"] is not None
    assert res["world_width"] > 0

    # Verify the produced absolute DSM is close to the true elevation
    import rasterio

    with rasterio.open(res["dsm_geotiff_path"]) as src:
        dsm = src.read(1).astype("float32")
    assert src.crs.to_string() == "EPSG:32633"
    # Regression: calibrated values should be in plausible meter range
    assert np.median(dsm) > 100.0  # heights are metric, not 0..1


def test_validation_endpoint(geo_environment, monkeypatch):
    upload, true_elev = geo_environment
    monkeypatch.setattr(depth, "infer_relative_dsm", _fake_rdsm)
    job = db.create_job(upload["id"], "absolute_dsm")
    jobs._run_job(job["id"])
    assert db.get_job(job["id"])["status"] == "done"

    data = jobs.run_validation(job["id"])
    assert data["rmse"] >= 0.0
    assert data["mae"] >= 0.0
    assert -1.0 <= data["correlation"] <= 1.0
    assert data["diff_heatmap_url"].endswith(".png")
    assert data["landscape_type"] is None
    assert data["reference_source"] == "local"

    res = db.get_result_by_job(job["id"])
    evs = db.get_evaluations_for_result(res["id"])
    assert len(evs) == 1
    assert evs[0]["landscape_type"] is None
    assert evs[0]["reference_source"] == "local"
    assert evs[0]["rmse"] == data["rmse"]


def test_validation_save_artifacts(geo_environment, monkeypatch):
    """/validate?save_artifacts=true exports the final persisted absolute DSM
    (never re-infers) without changing the reported metrics."""
    upload, true_elev = geo_environment
    monkeypatch.setattr(depth, "infer_relative_dsm", _fake_rdsm)
    job = db.create_job(upload["id"], "absolute_dsm")
    jobs._run_job(job["id"])
    assert db.get_job(job["id"])["status"] == "done"

    plain = jobs.run_validation(job["id"])
    data = jobs.run_validation(job["id"], save_artifacts=True)

    for k in ("rmse", "mae", "correlation", "held_out_pixel_count"):
        assert data[k] == plain[k]

    res = db.get_result_by_job(job["id"])
    created = [p for p in config.OUTPUT_DIR.rglob("predicted_depth.npy")]
    assert created, "no artifact directory was written"
    dir_ = created[0].parent
    # /validate only has the persisted absolute DSM: final-only export, no
    # raw/relative stages, no re-inference.
    for f in ("predicted_depth.npy", "predicted_depth.png",
              "predicted_depth.tif", "calibrated_prediction.npy",
              "metrics.json"):
        assert (dir_ / f).is_file(), f
    assert not (dir_ / "raw_prediction.npy").exists()
    assert not (dir_ / "relative_prediction.npy").exists()

    held_path = Path(res["heightmap_path"]).parent / "held_out.npz"
    assert held_path.is_file()  # untouched by artifact export


def test_validation_scores_held_out_not_all_pixels(geo_environment, monkeypatch):
    """/validate must have the same leak-proof guarantee as /evaluate: the
    reported numbers come from the persisted held-out 20% (never the 80% the
    calibration fit on), and the response exposes the held-out pixel count so
    the whole-grid (leaky) comparison is provably NOT what's being reported."""
    upload, true_elev = geo_environment
    monkeypatch.setattr(depth, "infer_relative_dsm", _fake_rdsm)
    job = db.create_job(upload["id"], "absolute_dsm")
    jobs._run_job(job["id"])
    assert db.get_job(job["id"])["status"] == "done"

    res = db.get_result_by_job(job["id"])
    held_path = Path(res["heightmap_path"]).parent / "held_out.npz"
    assert held_path.is_file(), "calibration must persist the holdout set"

    # Independent recomputation from the persisted set (mirrors test_evaluation's
    # test_calibration_leakage_*: same seed 42 / 80-20 split contract).
    fit = calib.load_fit(held_path)
    expected = compute_metrics(fit.scale * fit.held_relative + fit.offset,
                               fit.held_reference)

    data = jobs.run_validation(job["id"])
    assert data["held_out_pixel_count"] == pytest.approx(0.2 * fit.n_samples, abs=1)
    assert data["rmse"] == pytest.approx(expected["rmse"], rel=1e-6)
    assert data["mae"] == pytest.approx(expected["mae"], rel=1e-6)
    assert data["correlation"] == pytest.approx(expected["correlation"])
    assert data["held_out_pixel_count"] < fit.n_samples  # never the full grid
