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


def test_fit_affine_flags_chilika_like_low_relief_as_degenerate():
    """Chilika Lake (real tile, Odisha): the relative depth has real variance
    (so the legacy np.ptp(d_train)==0 check never fires), but the true terrain
    is 93-94% a flat 0-5 m lake floor with only a thin ~11% hill strip rising
    to ~950 m. RANSAC locks onto the flat majority and collapses to scale ~0,
    which previously exported an all-black heightmap and called it a texture.
    The relative low-relief guard must flag it degenerate WITHOUT rejecting the
    job: fit completes with a flat elevation, caller keeps running."""
    rng = np.random.default_rng(0)

    # Chilika-like ground truth: 93% lake floor 0-5 m, thin west hill strip
    # (~0.11 of the width, reaching ~950 m - the tile's real max).
    h, w = 400, 600
    truth = rng.uniform(0.0, 5.0, (h, w))
    strip = max(1, int(w * 0.11))
    ramp = np.linspace(0.0, 945.0, strip)[None, :]
    truth[:, :strip] = np.maximum(truth[:, :strip], np.broadcast_to(ramp, (h, strip)))

    # GAN-like relative depth: real variance, mean ~0.2 / std ~0.14 (matches the
    # Chilika /evaluate reproduction metrics), so this is NOT the legacy
    # constant-prediction degenerate case - it exercises the new low-relief guard.
    rdsm = np.clip(rng.normal(0.20, 0.14, (h, w)), 0.0, 1.0)

    fit = calib.fit_affine(rdsm, truth)
    assert fit.degenerate is True
    assert fit.method == "degenerate_low_relief"
    assert fit.scale == 0.0
    assert calib.degenerate_reason(fit) is not None

    # The job does NOT fail: apply_fit still runs and yields a flat surface
    # (the lake is genuinely flat), never a 500 or a silent garbage heightmap.
    out = calib.apply_fit(rdsm, fit)
    assert out.shape == rdsm.shape
    assert float(np.ptp(out)) == 0.0
    # the flat result sits on the lake floor (~0-5 m), not on a hill band
    assert 0.0 < fit.offset < 5.0


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


def test_ransac_sign_flip_falls_back_to_ols_scale():
    """RANSAC can lock onto a tight minority sub-population with a NEGATIVE
    slope while the global (train-split) trend is POSITIVE - the sparse_01
    pathology (raw corr +0.585, RANSAC scale -159). The fit must then use the
    OLS scale of the full train split (positive), keep it un-flipped, and flag
    the instability as ransac_sign_disagreement instead of emitting a negative
    calibration that would fake a REVERSED verdict."""
    rng = np.random.default_rng(7)
    n = 4096
    d = rng.uniform(0, 1, n)
    is_band = rng.random(n) < 0.85
    h = np.where(is_band,
                 -400.0 * d + 350.0 + rng.normal(0, 2.0, n),
                 3000.0 * d + 1000.0 + rng.normal(0, 2.0, n))
    fit = calib.fit_affine(d.reshape(64, 64), h.reshape(64, 64))
    assert fit.scale > 0.0
    assert fit.calibration_warning == "ransac_sign_disagreement"
    assert fit.polarity_inverted is False


def test_genuinely_inverted_keeps_negative_scale_and_flags_polarity():
    """A genuinely inverted model output (train-split raw correlation clearly
    negative) keeps its negative scale - never abs()'d - and is flagged
    polarity_inverted=True, because the inversion is real, not a fit artifact."""
    rng = np.random.default_rng(3)
    d = rng.uniform(0, 1, (64, 64))
    h = -200.0 * d + 400.0 + rng.normal(0, 20.0, d.shape)
    fit = calib.fit_affine(d, h)
    assert fit.scale < 0.0
    assert fit.calibration_warning == "negative_scale"
    assert fit.polarity_inverted is True
    assert "inverted relative to the reference" in fit.polarity_reason
