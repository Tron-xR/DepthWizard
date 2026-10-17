"""Phase 0 spatial-correctness checks: downsample footprint/float/nodata,
reference-DEM reprojection, export metadata, and the derived calibration
estimate status.

Invariants pinned here:
  - a downsampled working raster keeps the ORIGINAL footprint (bounds) with a
    rescaled affine, keeps float32 values (never an 8-bit quantize), and keeps
    its declared nodata;
  - a reference DEM is reprojected onto the exact working grid from its own
    CRS/transform, with the shape-only resize kept only as the no-georef
    fallback;
  - calibration exposes a derived estimate_status (valid / low_confidence /
    failed, "relative" when there is no DEM georeferencing) plus the RANSAC
    inlier count, on both /validate and /evaluate;
  - export_geotiff round-trips CRS/transform/bounds/nodata.
"""
import numpy as np
import pytest
from rasterio.transform import from_origin

from app.pipeline import (calibration as calib, dem_source, evaluation as ev,
                          exporter, uploader)


def _write_float_geotiff(path, data, transform, crs="EPSG:32633", nodata=None):
    import rasterio

    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    return path


# ---- downsample: footprint + float32 + nodata ------------------------------ #
def test_downsample_keeps_footprint_float32_and_nodata(tmp_path):
    orig_h, orig_w, cell = 1200, 1500, 10.0
    transform = from_origin(500000.0, 4650000.0, cell, cell)
    ramp_cols = np.linspace(100.0, 200.0, orig_w, dtype="float32")
    elev = np.tile(ramp_cols, (orig_h, 1))
    elev[0, 0] = -9999.0
    src = _write_float_geotiff(tmp_path / "big.tif", elev, transform, nodata=-9999.0)

    info = uploader.ingest_upload(src.read_bytes(), "big.tif")
    assert info["resample_note"] == "downsampled"
    assert info["bounds"] == pytest.approx(
        [500000.0, 4650000.0 - orig_h * cell, 500000.0 + orig_w * cell, 4650000.0])

    import rasterio

    with rasterio.open(info["file_path"]) as w:
        assert (w.height, w.width) == (819, 1024)
        # footprint (bounds) survives the downsample exactly
        assert w.bounds == pytest.approx(info["bounds"])
        assert w.crs.to_string() == "EPSG:32633"
        assert w.dtypes[0] == "float32"
        assert w.nodata == -9999.0
        # pixel size rescaled by the downsample ratio
        assert w.transform.a == pytest.approx(cell * orig_w / 1024)
        assert w.transform.e == pytest.approx(-cell * orig_h / 819)
        data = w.read(1)

    # float-preserving: a bilinear resize of a linear ramp reproduces the plane
    # (the previous uint8 quantize path had a ~0.2 m half-step at this range).
    interior = data[300:500, 10:1000]
    assert np.isfinite(interior).all()
    x_src = (np.arange(10, 1000) + 0.5) * orig_w / 1024 - 0.5
    expected = np.interp(x_src, np.arange(orig_w), ramp_cols)
    assert np.max(np.abs(interior - expected[None, :])) < 0.05


def test_downsample_masks_nan_and_nodata():
    orig_h, orig_w = 1200, 1500
    ramp_cols = np.linspace(100.0, 200.0, orig_w, dtype="float32")
    arr = np.tile(ramp_cols, (orig_h, 1))[None].copy()
    arr[0, 100:400, 200:500] = np.nan         # NaN block (source px)
    arr[0, 700:1000, 900:1200] = -9999.0      # declared-nodata block

    out, _ = uploader._downsample_array(arr, 1024, nodata=-9999.0)
    band, sy, sx = out[0], out.shape[1] / orig_h, out.shape[2] / orig_w
    assert out.dtype == np.float32

    # invalid regions resample as the DECLARED nodata value (kept numeric) and
    # stay confined to their mapped source footprint + half-pixel rim
    invalid = band == -9999.0
    assert np.isnan(band).sum() == 0
    assert invalid[int(250 * sy), int(350 * sx)] and invalid[int(850 * sy), int(1050 * sx)]
    rows = np.where(invalid[:, int(350 * sx)])[0]
    assert rows.min() >= int(100 * sy) - 1 and rows.max() <= int(400 * sy) + 1
    cols = np.where(invalid[int(250 * sy), :])[0]
    assert cols.min() >= int(200 * sx) - 1 and cols.max() <= int(500 * sx) + 1
    rows2 = np.where(invalid[:, int(1050 * sx)])[0]
    assert rows2.min() >= int(700 * sy) - 1 and rows2.max() <= int(1000 * sy) + 1
    cols2 = np.where(invalid[int(850 * sy), :])[0]
    assert cols2.min() >= int(900 * sx) - 1 and cols2.max() <= int(1200 * sx) + 1
    assert not invalid[277:474].any() and not invalid[:, 345:610].any()

    # valid pixels are the weighted mean of VALID samples only: the ramp is
    # reproduced to sub-quantization tolerance (no -9999 smearing)
    row = band[350]
    x_src = (np.arange(band.shape[1]) + 0.5) / sx - 0.5
    expected = np.interp(x_src, np.arange(orig_w), ramp_cols)
    assert np.max(np.abs(row - expected)) < 0.05

    # without a declared nodata value NaN stays NaN (pipeline convention)
    # instead of smearing into valid neighbours, while -9999 is just data
    nband = uploader._downsample_array(arr, 1024)[0][0]
    assert np.isnan(nband[int(250 * sy), int(350 * sx)])
    assert not np.isnan(nband[int(850 * sy), int(1050 * sx)])
    nrows, ncols = np.where(np.isnan(nband))
    assert nrows.min() >= int(100 * sy) - 1 and nrows.max() <= int(400 * sy) + 1
    assert ncols.min() >= int(200 * sx) - 1 and ncols.max() <= int(500 * sx) + 1


def test_small_upload_keeps_transform_untouched(tmp_path):
    transform = from_origin(500000.0, 4650000.0, 30.0, 30.0)
    elev = np.linspace(10, 20, 100 * 80, dtype="float32").reshape(100, 80)
    src = _write_float_geotiff(tmp_path / "small.tif", elev, transform)

    info = uploader.ingest_upload(src.read_bytes(), "small.tif")
    assert info["resample_note"] == "as-is"

    import rasterio

    with rasterio.open(info["file_path"]) as w:
        assert (w.height, w.width) == (100, 80)
        assert w.transform == transform
        assert list(w.bounds) == pytest.approx(info["bounds"])


def test_read_local_geotiff_exposes_georef_and_nanifies_nodata(tmp_path):
    transform = from_origin(500000.0, 4650000.0, 30.0, 30.0)
    data = np.full((4, 4), 50.0, dtype="float32")
    data[0, 0] = -9999.0
    path = _write_float_geotiff(tmp_path / "ref.tif", data, transform,
                                nodata=-9999.0)

    arr, meta = dem_source._read_local_geotiff(str(path))
    assert arr.dtype == np.float32
    assert np.isnan(arr[0, 0]) and arr[1, 1] == 50.0
    assert meta["crs"] == "EPSG:32633"
    assert meta["transform"] == transform
    assert (meta["width"], meta["height"]) == (4, 4)


# ---- reference DEM: true reprojection, not an array resize ----------------- #
def test_resample_reference_reprojects_onto_target_grid():
    src_transform = from_origin(499000.0, 4651000.0, 100.0, 100.0)
    src_h, src_w = 30, 30
    yy, xx = np.mgrid[0:src_h, 0:src_w].astype("float64")
    world_x = src_transform.c + (xx + 0.5) * src_transform.a
    world_y = src_transform.f + (yy + 0.5) * src_transform.e
    src = (200.0 + 0.05 * (world_x - 499000.0)
           + 0.03 * (4651000.0 - world_y)).astype("float32")
    meta = {"source": "test", "crs": "EPSG:32633", "transform": src_transform,
            "width": src_w, "height": src_h}

    tgt_h, tgt_w = 8, 8
    tgt_transform = from_origin(500000.0, 4650000.0, 150.0, 150.0)
    out = calib.resample_reference(src, meta, (tgt_h, tgt_w), "EPSG:32633",
                                   tgt_transform)

    assert out.shape == (tgt_h, tgt_w)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    ty, tx = np.mgrid[0:tgt_h, 0:tgt_w].astype("float64")
    expected = (200.0
                + 0.05 * (tgt_transform.c + (tx + 0.5) * tgt_transform.a - 499000.0)
                + 0.03 * (4651000.0 - (tgt_transform.f + (ty + 0.5) * tgt_transform.e)))
    # True geolocation: value == the source plane evaluated at the target pixel
    # centers, to within GDAL's sub-pixel warp approximation (~0.03 source px =
    # 0.22 m here). The old blind array-resize path maps FRACTIONALLY instead
    # of by extent, erring by tens of meters on this geometry, so 0.5 m
    # discriminates real reprojection cleanly.
    assert np.max(np.abs(out - expected)) < 0.5


def test_resample_reference_falls_back_without_georeferencing():
    src = np.linspace(0, 10, 64, dtype="float32").reshape(8, 8)
    meta = {"source": "test", "crs": None, "transform": None,
            "width": 8, "height": 8}
    out = calib.resample_reference(src, meta, (5, 5), None, None)
    np.testing.assert_array_equal(out, calib.resample_to_grid(src, (5, 5)))


# ---- export metadata round trip -------------------------------------------- #
def test_export_geotiff_round_trips_nodata(tmp_path):
    transform = from_origin(500000.0, 4650000.0, 30.0, 30.0)
    bounds = [500000.0, 4649700.0, 500300.0, 4650000.0]
    elev = np.arange(100, dtype="float32").reshape(10, 10)
    elev[0, 0] = -9999.0
    out = tmp_path / "dsm.tif"

    exporter.export_geotiff(elev, "EPSG:32633", transform, bounds, out,
                            nodata=-9999.0)

    import rasterio

    with rasterio.open(out) as src:
        assert src.crs.to_string() == "EPSG:32633"
        assert src.transform == transform
        assert list(src.bounds) == pytest.approx(bounds)
        assert (src.height, src.width) == (10, 10)
        assert src.dtypes[0] == "float32"
        assert src.nodata == -9999.0
        assert src.read(1)[0, 0] == -9999.0


def test_export_dsm_tags_estimate_status(tmp_path):
    transform = from_origin(0.0, 10.0, 1.0, 1.0)
    src = _write_float_geotiff(tmp_path / "dsm.tif", np.ones((10, 10)),
                               transform, crs="EPSG:32633")
    out = exporter.export_dsm_geotiff(
        src, tmp_path / "dsm_export.tif", job_id="j1", input_filename="in.tif",
        backend="test_backend", model="m1", min_elev=1.0, max_elev=1.0,
        estimate_status="low_confidence")

    import rasterio

    with rasterio.open(out) as ds:
        assert ds.tags()["ESTIMATE_STATUS"] == "low_confidence"


# ---- derived estimate status ----------------------------------------------- #
def test_estimate_status_maps_health_to_three_levels(tmp_path):
    ramp = np.linspace(0, 1, 64 * 64).reshape(64, 64)

    ok = calib.fit_affine(ramp, 5.0 + 10.0 * ramp)
    assert ok.calibration_warning is None
    assert ok.estimate_status == "valid"

    flagged = calib.fit_affine(ramp, 120.0 - 100.0 * ramp)
    assert flagged.calibration_warning is not None
    assert flagged.estimate_status == "low_confidence"

    failed = calib.fit_affine(np.full((32, 32), 0.5),
                              np.linspace(0, 1, 32 * 32).reshape(32, 32))
    assert failed.degenerate is True
    assert failed.estimate_status == "failed"

    # derived, not stored: a save/load round trip keeps the same status
    p = tmp_path / "held_out.npz"
    calib.save_fit(flagged, p)
    assert calib.load_fit(p).estimate_status == "low_confidence"


def test_too_few_inliers_warns(monkeypatch):
    class _FakeRansac:
        def __init__(self, **_kw):
            pass

        def fit(self, X, y):
            self.inlier_mask_ = np.zeros(len(y), dtype=bool)
            self.inlier_mask_[:8] = True
            return self

    monkeypatch.setattr(calib, "RANSACRegressor", _FakeRansac)
    ramp = np.linspace(0, 1, 160).reshape(10, 16)
    fit = calib.fit_affine(ramp, 5.0 + 10.0 * ramp)
    # 8 RANSAC inliers of 128 training pixels (< 10%): extrapolation, not
    # calibration, so the fit must warn and drop to low_confidence.
    assert fit.calibration_warning == "too_few_inliers"
    assert fit.estimate_status == "low_confidence"

    # the threshold is configuration (config.CALIB_MIN_INLIER_FRACTION),
    # not a magic number inside the fit
    monkeypatch.setattr(calib.config, "CALIB_MIN_INLIER_FRACTION", 0.0)
    assert calib.fit_affine(ramp, 5.0 + 10.0 * ramp).calibration_warning is None


# ---- /evaluate surface ----------------------------------------------------- #
def test_evaluate_exposes_status_stats_and_schema():
    from app.schemas import EvaluationResponse

    ramp = np.linspace(0, 1, 32 * 32).reshape(32, 32).astype("float64")
    res = ev.evaluate_prediction_truth(ramp, 5.0 + 10.0 * ramp, mode="calibrated")

    assert res["estimate_status"] == "valid"
    assert res["calibration_inlier_count"] is not None
    assert res["bias"] == pytest.approx(0.0, abs=1e-6)
    assert res["prediction_std"] == pytest.approx(res["reference_std"], rel=1e-3)
    EvaluationResponse(**res)  # the response route forwards every field

    # non-georeferenced evaluations report the "relative" taxonomy status;
    # uncallibratable absolute branches stay None instead of fabricating one
    rel = ev.evaluate_prediction_truth(np.ones((16, 16)),
                                       np.linspace(0, 50, 256).reshape(16, 16),
                                       mode="relative")
    assert rel["estimate_status"] == "relative"
    assert rel["bias"] is None
    assert rel["calibration_inlier_count"] is None
