"""Tests for the ad hoc RGB + ground-truth DEM evaluation (POST /evaluate).

Covers the metric math, nodata/NaN masking, constant-array edge case, alignment
refusal without georeferencing, a full HTTP integration flow, and the
calibration-leakage guarantee (held-out 20% scored separately from the fit).
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.pipeline import calibration as calib, depth, evaluation
from app.pipeline.validate import compute_metrics
from tests.conftest import make_rgb_gradient, write_geotiff, write_rgb_png


def _grid(dem, rdsm, *, nodata=None, mode="calibrated"):
    return evaluation.evaluate_prediction_truth(rdsm, dem, nodata=nodata, mode=mode)


# --------------------------------------------------------------------------- #
# A. Perfect prediction
# --------------------------------------------------------------------------- #
def test_perfect_prediction():
    rdsm = np.random.default_rng(0).random((128, 128))
    dem = 3.0 * rdsm + 10.0
    res = _grid(dem, rdsm)
    assert res["calibrated"] is True
    assert res["mae"] == pytest.approx(0.0, abs=1e-6)
    assert res["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert res["correlation"] == pytest.approx(1.0, abs=1e-9)
    assert res["held_out_pixel_count"] == pytest.approx(0.2 * res["valid_pixels"], abs=1)


# --------------------------------------------------------------------------- #
# B. Constant error (+5). The calibrated endpoint absorbs the offset as its
#    scale/offset fit, so the constant-offset identity is asserted at the raw
#    reused-metric level and via relative mode.
# --------------------------------------------------------------------------- #
def test_constant_offset_raw_metrics_and_relative_mode():
    rdsm = np.linspace(0, 1, 20 * 20).reshape(20, 20)
    dem = rdsm + 5.0

    raw = compute_metrics(rdsm, dem)  # reused canonical metric function
    assert raw["mae"] == pytest.approx(5.0)
    assert raw["rmse"] == pytest.approx(5.0)
    assert raw["correlation"] == pytest.approx(1.0, abs=1e-9)

    res = _grid(dem, rdsm, mode="relative")
    assert res["mae"] is None and res["rmse"] is None
    assert res["reason"] == "relative depth, uncalibrated"
    assert res["correlation"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# C. Large outlier: RMSE reacts more strongly than MAE
# --------------------------------------------------------------------------- #
def test_large_outlier_hits_rmse_harder_than_mae():
    rdsm = np.linspace(0, 1, 10000).reshape(100, 100)
    dem = 100.0 * rdsm + 50.0
    dem.flat[0] = 1e6  # single gross outlier
    raw = compute_metrics(rdsm, dem)
    assert raw["rmse"] > raw["mae"] * 10


# --------------------------------------------------------------------------- #
# D. NaN / Inf / DEM nodata are excluded, and the count is reported
# --------------------------------------------------------------------------- #
def test_nan_and_nodata_excluded():
    rdsm = np.ones((10, 10))
    dem = 3.0 * rdsm + 1.0
    rdsm[0, 0] = np.nan
    dem[1, 1] = np.inf
    dem[2, 2] = -9999.0
    res = _grid(dem, rdsm, nodata=-9999.0)
    assert res["total_pixels"] == 100
    assert res["valid_pixels"] == 100 - 3


def test_all_invalid_raises():
    rdsm = np.full((4, 4), np.nan)
    dem = np.ones((4, 4))
    with pytest.raises(ValueError, match="No valid pixels"):
        _grid(dem, rdsm)


# --------------------------------------------------------------------------- #
# E. Constant arrays: correlation must be null, no crash
# --------------------------------------------------------------------------- #
def test_constant_arrays_correlation_none():
    res = _grid(np.ones((20, 20)), np.ones((20, 20)))
    assert res["correlation"] is None
    assert res["mae"] is None and res["rmse"] is None
    assert res["calibrated"] is False


def test_degenerate_calibration_falls_back_to_relative():
    """Prediction varies ONLY on pixels that land in the held-out 20% (by the
    fixed split), so the training 80% has zero variance: calibration is
    impossible and must degrade to a relative correlation, never a 500."""
    rdsm = np.zeros((128, 128))
    order = np.random.default_rng(calib._SPLIT_SEED).permutation(rdsm.size)
    n_tr = int(round(rdsm.size * calib._TRAIN_FRACTION))
    rdsm.flat[order[n_tr:]] = np.random.default_rng(1).random(int(rdsm.size - n_tr))
    dem = 50.0 * rdsm + 10.0
    res = _grid(dem, rdsm)
    assert res["mae"] is None and res["rmse"] is None
    assert res["calibrated"] is False
    assert "degenerate" in res["reason"]
    assert res["correlation"] > 0.99  # shape signal survives the fallback


# --------------------------------------------------------------------------- #
# F. Different dimensions: refuse without spatial metadata, pass through when
#    equal.
# --------------------------------------------------------------------------- #
def test_align_equal_shapes_passthrough():
    pred = np.zeros((4, 4))
    dem = np.ones((4, 4))
    out = evaluation.align_dem_to_prediction(pred, dem)
    np.testing.assert_array_equal(out, dem)


def test_align_mismatch_refuses_without_georef():
    pred = np.zeros((4, 4))
    dem = np.ones((8, 8))
    with pytest.raises(ValueError, match="pixel-aligned"):
        evaluation.align_dem_to_prediction(pred, dem)


def test_georef_warp_sets_uncovered_region_to_nodata():
    """Areas outside the source DEM extent must become NaN (excluded by the
    valid mask), never a fabricated 0 elevation."""
    from affine import Affine

    pred = np.zeros((4, 8))
    dem = np.arange(16, dtype="float32").reshape(4, 4)
    dst = evaluation.align_dem_to_prediction(
        pred, dem,
        # Same row span, but the prediction grid spans x in [0,8) while the
        # source DEM only covers [0,4): the right half has no source data.
        predicted_transform=Affine.translation(0, 4) * Affine.scale(1.0, -1.0),
        predicted_crs="EPSG:3857",
        dem_transform=Affine.translation(0, 4) * Affine.scale(1.0, -1.0),
        dem_crs="EPSG:3857",
    )
    assert dst.shape == (4, 8)
    # The source data that DOES fall inside the prediction extent survives...
    assert np.isfinite(dst[:, :4]).all()
    # ...and the uncovered columns become nodata, excluded by the valid mask.
    assert float((~np.isfinite(dst[:, 4:])).sum()) == 4 * 4
    valid = evaluation.build_valid_mask(pred, dst)
    assert valid.sum() == 4 * 4


# --------------------------------------------------------------------------- #
# G. HTTP integration: upload a synthetic RGB + matches its DEM, evaluate
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(monkeypatch):
    def fake_infer(rgb, **kwargs):
        h, w = np.asarray(rgb).shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype("float32")
        return np.asarray(xx / max(w - 1, 1) * 0.5 + yy / max(h - 1, 1) * 0.25,
                          dtype="float32")

    monkeypatch.setattr(depth, "infer_relative_dsm", fake_infer)
    with TestClient(app) as c:
        yield c


def test_evaluate_endpoint_full_flow(client, tmp_path):
    size = 96
    rgb = make_rgb_gradient(size, size)
    image_path = write_rgb_png(tmp_path / "rgb.png", rgb)
    yy, xx = np.mgrid[0:size, 0:size].astype("float64")
    rdsm = xx / max(size - 1, 1) * 0.5 + yy / max(size - 1, 1) * 0.25
    truth = 120.0 * rdsm + 30.0
    dem_path = write_geotiff(tmp_path / "truth.tif", truth, cell=30.0)

    with open(image_path, "rb") as fimg, open(dem_path, "rb") as fdem:
        resp = client.post(
            "/evaluate",
            files={"image": ("rgb.png", fimg.read(), "image/png"),
                   "dem": ("truth.tif", fdem.read(), "image/tiff")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["calibrated"] is True
    assert body["valid_pixels"] == size * size
    assert body["total_pixels"] == size * size
    assert body["held_out_pixel_count"] == pytest.approx(0.2 * size * size, abs=1)
    assert body["mae"] < 1.0
    assert body["rmse"] < 1.0
    assert body["correlation"] > 0.99


def test_evaluate_rejects_bad_mode(client, tmp_path):
    rgb = make_rgb_gradient(32, 32)
    image_path = write_rgb_png(tmp_path / "rgb.png", rgb)
    dem_path = write_geotiff(tmp_path / "t.tif", np.ones((32, 32)))
    with open(image_path, "rb") as fimg, open(dem_path, "rb") as fdem:
        resp = client.post(
            "/evaluate",
            files={"image": ("rgb.png", fimg.read(), "image/png"),
                   "dem": ("t.tif", fdem.read(), "image/tiff")},
            data={"mode": "sideways"},
        )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "bad_request"


# --------------------------------------------------------------------------- #
# H. Calibration leakage: held-out RMSE must be measurably worse than a naive
#    all-pixel-fit RMSE - proving the 20% split is real, not a no-op.
# --------------------------------------------------------------------------- #
def test_calibration_leakage_held_out_worse_than_naive_all_fit():
    rdsm = np.random.default_rng(7).random((100, 100))
    n = rdsm.size

    # Replicate the deterministic split from calibration.py so we can put the
    # held-out pixels in a different elevation band than the fit pixels.
    order = np.random.default_rng(calib._SPLIT_SEED).permutation(n)
    n_train = int(round(n * calib._TRAIN_FRACTION))
    held_idx = order[n_train:]

    dem = 100.0 * rdsm + 20.0
    dem.flat[held_idx] += 60.0  # held-out 20% live 60 m above the fitted band

    res = _grid(dem, rdsm)
    assert res["calibrated"] is True
    assert res["held_out_pixel_count"] == pytest.approx(0.2 * n, abs=1)

    # Naive: fit affine on ALL pixels, score on ALL pixels (leaky).
    naive = calib.gcp_affine(rdsm.ravel(), dem.ravel())
    naive_pred = naive.scale * rdsm.ravel() + naive.offset
    naive_rmse = float(np.sqrt(np.mean((naive_pred - dem.ravel()) ** 2)))

    assert res["rmse"] > naive_rmse * 1.5, "held-out RMSE must be worse than a leaky all-pixel fit"


# --------------------------------------------------------------------------- #
# I. Opt-in comparison visualization (include_visualization)
# --------------------------------------------------------------------------- #
def _make_pair(tmp_path, size=64):
    rgb = make_rgb_gradient(size, size)
    image_path = write_rgb_png(tmp_path / "rgb.png", rgb)
    yy, xx = np.mgrid[0:size, 0:size].astype("float64")
    rdsm = xx / max(size - 1, 1) * 0.5 + yy / max(size - 1, 1) * 0.25
    truth = 120.0 * rdsm + 30.0
    dem_path = write_geotiff(tmp_path / "truth.tif", truth, cell=30.0)
    return image_path, dem_path


def _post_evaluate(client, image_path, dem_path, data=None):
    with open(image_path, "rb") as fimg, open(dem_path, "rb") as fdem:
        return client.post(
            "/evaluate",
            files={"image": ("rgb.png", fimg.read(), "image/png"),
                   "dem": ("truth.tif", fdem.read(), "image/tiff")},
            data=data or {},
        )


def test_default_response_comparison_image_is_null(client, tmp_path):
    """Opted-out call: the new field is present but null (matches the schema's
    Optional[...]=None convention) and every existing field is untouched."""
    img, dem = _make_pair(tmp_path)
    resp = _post_evaluate(client, img, dem)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["comparison_image_url"] is None
    # existing behavior unchanged
    assert body["calibrated"] is True
    assert body["mae"] is not None
    assert body["held_out_pixel_count"] > 0


def test_visualization_true_saves_and_serves_file(client, tmp_path):
    img, dem = _make_pair(tmp_path)
    resp = _post_evaluate(client, img, dem, data={"include_visualization": "true"})
    assert resp.status_code == 200, resp.text
    url = resp.json()["comparison_image_url"]
    assert url and url.startswith("/files/") and url.endswith("comparison.png")
    name = url.split("/")[2]
    saved = config.FILES_DIR / name / "comparison.png"
    assert saved.is_file(), "visualization must exist on disk after the call"
    served = client.get(url)
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/png")


def _panel(arr, i, w):
    x = i * (w + evaluation._PANEL_GAP)
    return arr[evaluation._LABEL_H:, x:x + w]


def test_comparison_panels_share_color_scale(tmp_path):
    """Ground-truth and predicted panels use ONE normalization: a predicted
    surface 1000 m higher than truth must NOT render identically (auto-scaling
    each panel would make the bad prediction look like the truth)."""
    from PIL import Image as PILImage

    truth = np.random.default_rng(4).random((32, 32)) * 10.0    # 0..10 m
    predicted = truth + 1000.0                                  # 1000..1010 m
    rgb = make_rgb_gradient(32, 32)

    out = evaluation.render_height_comparison(rgb, truth, predicted, tmp_path / "cmp.png")
    arr = np.asarray(PILImage.open(tmp_path / "cmp.png").convert("RGB")).astype("int64")
    truth_panel = _panel(arr, 1, 32)
    pred_panel = _panel(arr, 2, 32)
    assert not np.array_equal(truth_panel, pred_panel), \
        "independent scaling would render these panels identically; they must share (vmin, vmax)"

    evaluation.render_height_comparison(rgb, truth, truth, tmp_path / "same.png")
    arr2 = np.asarray(PILImage.open(tmp_path / "same.png").convert("RGB")).astype("int64")
    np.testing.assert_array_equal(_panel(arr2, 1, 32), _panel(arr2, 2, 32),
                                  "identical inputs must render identical panels")