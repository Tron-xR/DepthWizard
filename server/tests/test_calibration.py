"""Unit tests for the scale-calibration math (Epic 3, 09-tech-stack RANSAC fit)."""
import numpy as np
import pytest

from app.pipeline import calibration as calib


def test_fit_affine_recovers_exact_scale_offset():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 1, size=(64, 64))
    true_scale, true_off = 180.0, 25.0
    h = d * true_scale + true_off
    fit = calib.fit_affine(d, h)
    assert fit.scale == pytest.approx(true_scale, abs=0.5)
    assert fit.offset == pytest.approx(true_off, abs=0.5)


def test_fit_affine_robust_to_outliers():
    rng = np.random.default_rng(1)
    d = rng.uniform(0, 1, size=(80, 80))
    true_scale, true_off = 120.0, 10.0
    h = d * true_scale + true_off
    # Add 20% outliers (e.g. building/vegetation spikes)
    outlier_mask = rng.random(d.shape) < 0.2
    h = h + outlier_mask * rng.uniform(-500, 500, size=h.shape).astype("float32")
    fit = calib.fit_affine(d, h)
    assert fit.scale == pytest.approx(true_scale, abs=20.0)
    assert fit.offset == pytest.approx(true_off, abs=30.0)
    assert fit.n_inliers > 0.6 * fit.n_samples


def test_apply_fit():
    d = np.linspace(0, 1, 25).reshape(5, 5)
    fit = calib.CalibrationFit(scale=100, offset=50, n_samples=25,
                               n_inliers=25, rmse=0, method="gcp_affine")
    out = calib.apply_fit(d, fit)
    assert out[0, 0] == pytest.approx(50.0)
    assert out[-1, -1] == pytest.approx(150.0)


def test_gcp_affine():
    d = np.array([0.0, 0.5, 1.0])
    h = np.array([20.0, 120.0, 220.0])
    fit = calib.gcp_affine(d, h)
    assert fit.scale == pytest.approx(200.0)
    assert fit.offset == pytest.approx(20.0)


def test_not_enough_samples():
    d = np.array([[0.1], [0.2]])
    h = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError):
        calib.fit_affine(d, h)


def test_fit_affine_holds_out_20_percent():
    rng = np.random.default_rng(5)
    d = rng.uniform(0, 1, size=(64, 64))
    true_scale, true_off = 180.0, 25.0
    h = d * true_scale + true_off
    fit = calib.fit_affine(d, h)
    # exactly ~20% of valid samples reserved as held-out validation data
    assert len(fit.held_relative) == pytest.approx(0.2 * fit.n_samples, abs=1)
    assert len(fit.held_reference) == len(fit.held_relative)
    # held-out pixels lie on the exact line -> fit extrapolates with zero error
    resid = (fit.scale * fit.held_relative + fit.offset) - fit.held_reference
    assert np.abs(resid).max() < 1e-6


def test_resample_to_grid_shapes():
    src = np.arange(16, dtype="float32").reshape(4, 4)
    out = calib.resample_to_grid(src, (8, 8))
    assert out.shape == (8, 8)
    assert out.dtype == np.float32


def test_save_load_fit_round_trip(tmp_path):
    """The held-out set persisted for /validate must round-trip losslessly."""
    rng = np.random.default_rng(7)
    d = rng.uniform(0, 1, (48, 48))
    h = d * 90.0 + 12.0
    fit = calib.fit_affine(d, h)
    p = tmp_path / "held_out.npz"
    calib.save_fit(fit, p)
    loaded = calib.load_fit(p)
    assert loaded.scale == fit.scale
    assert loaded.offset == fit.offset
    assert loaded.method == fit.method
    assert loaded.degenerate is False
    np.testing.assert_array_equal(loaded.held_relative, fit.held_relative)
    np.testing.assert_array_equal(loaded.held_reference, fit.held_reference)
    # legacy jobs simply have no file -> None, never a crash
    assert calib.load_fit(tmp_path / "missing.npz") is None
