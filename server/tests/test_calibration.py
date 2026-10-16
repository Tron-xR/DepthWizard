"""Unit tests for the scale-calibration math (Epic 3, 09-tech-stack RANSAC fit)."""
from pathlib import Path

import numpy as np
import pytest

from app.pipeline import calibration as calib

REPO_ROOT = Path(__file__).resolve().parents[2]
_PIX2PIX_PREDICTIONS = (REPO_ROOT / "benchmark_runs" / "c3a2841_pix2pix_20261016T134335Z"
                        / "predictions")


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
    assert loaded.ransac_scale == fit.ransac_scale
    assert loaded.ols_scale == fit.ols_scale
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
    # warnings accumulate (single field, "; "-joined): the fixture is ALSO a
    # near-constant model output (raw std << reference std), so both hazards
    # are surfaced rather than the sign-disagreement shadowing the other
    assert fit.calibration_warning == "ransac_sign_disagreement; prediction_near_constant"
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
    # same accumulate semantics as the flip test: raw d (std 0.29) vs h (relief
    # ~400) makes the fixture near-constant too, surfaced alongside the sign warn
    assert fit.calibration_warning == "negative_scale; prediction_near_constant"
    assert fit.polarity_inverted is True
    assert "inverted relative to the reference" in fit.polarity_reason


def test_fit_affine_flags_absurd_magnitude_scale():
    """Symmetric opposite of the near-zero guard (job 8b1b017c shape): a fitted
    scale that maps the model's raw depth range onto a span > 2x the tile's own
    relief is amplification junk, not a real magnitude. It must surface as a
    warning - joined with the OTHER independent warnings on the same fit - while
    leaving the numerically-valid fit in place (nothing is rejected or zeroed)."""
    rng = np.random.default_rng(6)
    n = 64
    d = rng.normal(0.5, 0.02, (n, n))
    spike = rng.random((n, n)) < 0.1
    d[spike] += 4.0 * rng.normal(0, 1, int(spike.sum())).reshape(-1)
    noise = rng.normal(0, 1, (n, n))
    h = 2000.0 * np.sin(3 * np.pi * np.linspace(0, 1, n))[None, :] - 60.0 * (d - 0.5) + noise

    fit = calib.fit_affine(d, h)
    assert fit.degenerate is False
    assert fit.calibration_status == "warning"
    # the magnitude hazard is present, and the sign + flatness hazards of the
    # same fit are still surfaced instead of being shadowed
    assert fit.calibration_warning == "negative_scale; prediction_near_constant; scale_too_large"
    # sanity: the fixture really is the amplification pathology (span > 2x relief)
    amplification = abs(fit.scale) * float(np.ptp(d)) / float(np.ptp(h))
    assert amplification > 2.0


# ---------------------------------------------------------------------------
# Phase-3/4 additions: audit fields, determinism, holdout purity, and the
# diagnosed flip shapes (sign reconciliation + low-relief guard).
# ---------------------------------------------------------------------------
def test_fit_affine_deterministic_and_records_both_candidate_slopes():
    rng = np.random.default_rng(11)
    d = rng.uniform(0, 1, (64, 64))
    h = 300.0 * d + 10.0 + rng.normal(0, 5.0, d.shape)
    fit_a = calib.fit_affine(d, h)
    fit_b = calib.fit_affine(d, h)
    assert fit_a.scale == fit_b.scale
    assert fit_a.offset == fit_b.offset
    # the pre-decision RANSAC / OLS slopes are recorded on the fit and repeatable
    assert fit_a.ransac_scale is not None and fit_a.ols_scale is not None
    assert fit_a.ransac_scale == fit_b.ransac_scale
    assert fit_a.ols_scale == fit_b.ols_scale
    # clean data: both candidates agree in sign with the final scale
    assert fit_a.ransac_scale > 0.0 and fit_a.ols_scale > 0.0


def test_held_out_pixels_never_influence_the_fit():
    rng = np.random.default_rng(12)
    d = rng.uniform(0, 1, size=(64, 64))
    h = 150.0 * d + 25.0
    fit = calib.fit_affine(d, h)
    # reconstruct the fixed-seed split contract: corrupt ONLY the held-out 20%
    order = np.random.default_rng(calib._SPLIT_SEED).permutation(d.size)
    n_train = int(round(d.size * calib._TRAIN_FRACTION))
    h2 = h.ravel().copy()
    h2[order[n_train:]] += 1e6
    fit2 = calib.fit_affine(d, h2.reshape(h.shape))
    assert fit2.scale == fit.scale
    assert fit2.offset == fit.offset
    assert fit2.rmse == fit.rmse


def test_fit_affine_drops_nan_and_inf_pixels():
    rng = np.random.default_rng(13)
    d = rng.uniform(0, 1, (32, 32))
    h = 100.0 * d + 5.0
    d[0, :4] = np.nan
    h[1, :4] = np.inf
    h[2, :4] = np.nan
    fit = calib.fit_affine(d, h)
    assert fit.n_samples == int((np.isfinite(d) & np.isfinite(h)).sum())
    assert fit.scale == pytest.approx(100.0, rel=1e-6)


def test_sign_stable_under_pixel_permutation_and_float32():
    rng = np.random.default_rng(14)
    d = rng.uniform(0, 1, size=(64, 64))
    h = 220.0 * d + 30.0 + rng.normal(0, 3.0, d.shape)
    fit = calib.fit_affine(d, h)
    idx = rng.permutation(d.size)
    fit_p = calib.fit_affine(d.ravel()[idx].reshape(d.shape), h.ravel()[idx].reshape(h.shape))
    assert fit_p.scale > 0.0
    assert fit_p.scale == pytest.approx(fit.scale, rel=0.05)
    fit32 = calib.fit_affine(d.astype("float32"), h)
    assert fit32.scale > 0.0
    assert fit32.scale == pytest.approx(fit.scale, rel=1e-4)


def test_ransac_positive_ols_negative_keeps_negative_ols():
    """forested_10/11 shape (mirror of the sparse_01 fixture): RANSAC locks onto
    a tight POSITIVE sub-population while the full-train OLS is clearly NEGATIVE.
    The fit must keep the OLS sign (negative) and flag the disagreement - it must
    never force the scale positive to satisfy the brighter=higher convention."""
    rng = np.random.default_rng(8)
    n = 4096
    d = rng.uniform(0, 1, n)
    is_band = rng.random(n) < 0.8
    h = np.where(is_band,
                 400.0 * d + 350.0 + rng.normal(0, 2.0, n),      # tight positive band
                 -15000.0 * d + 12000.0 + rng.normal(0, 50.0, n))  # steep negative tail
    fit = calib.fit_affine(d.reshape(64, 64), h.reshape(64, 64))
    assert fit.ransac_scale > 0.0
    assert fit.ols_scale < 0.0
    assert fit.scale < 0.0
    assert "ransac_sign_disagreement" in fit.calibration_warning


def test_low_relief_guard_zeroes_tiny_scale_and_records_pre_guard_slopes():
    """sparse_17 shape: RANSAC+OLS agree on a tiny slope against a mostly-flat
    plateau with scattered ravines; the low-relief guard must return the flat
    degenerate fit (scale 0, median offset) while still recording the two
    pre-guard candidate slopes for the audit trail."""
    rng = np.random.default_rng(9)
    d = rng.uniform(0, 1, (64, 128))
    h = 1200.0 + rng.normal(0, 3.0, d.shape)
    ravine = rng.random(d.shape) < 0.25
    h[ravine] -= 300.0 * rng.random(int(ravine.sum()))
    fit = calib.fit_affine(d, h)
    assert fit.degenerate is True
    assert fit.method == "degenerate_low_relief"
    assert fit.scale == 0.0
    assert fit.calibration_warning == "near_zero_scale"
    assert fit.ransac_scale is not None and fit.ols_scale is not None
    assert calib.degenerate_reason(fit) is not None


@pytest.mark.skipif(not _PIX2PIX_PREDICTIONS.is_dir(),
                    reason="cached pix2pix benchmark predictions not present")
def test_seven_flip_tiles_replay_reproducibly():
    """Lock the diagnosed flip set against the real cached float32 predictions
    (2026-10-16 diagnostics): six sign-reconciliation tiles and one low-relief
    guard tile. Replaying the current fit_affine must be deterministic and
    reproduce the documented signs/degeneracy - if RANSAC, sklearn or the
    reconciliation logic drifts, this fails loudly."""
    import rasterio

    tiles_root = REPO_ROOT / "training_data"
    expected = {
        "forested_10": "reconciliation-negative", "forested_11": "reconciliation-negative",
        "hilly_07": "reconciliation-positive", "hilly_16": "reconciliation-positive",
        "sparse_01": "reconciliation-positive", "sparse_02": "reconciliation-positive",
        "sparse_17": "degenerate",
    }
    for tile_id, kind in expected.items():
        npy = _PIX2PIX_PREDICTIONS / f"{tile_id}_relative.npy"
        dem_paths = list(tiles_root.rglob(f"{tile_id}_dem.tif"))
        if not npy.is_file() or not dem_paths:
            pytest.skip(f"cached benchmark data for {tile_id} not present")
        rdsm = np.load(npy)
        with rasterio.open(dem_paths[0]) as src:
            dem = src.read(1).astype("float64")
            nodata = src.nodata
        assert rdsm.shape == dem.shape
        valid = np.isfinite(rdsm) & np.isfinite(dem)
        if nodata is not None and np.isfinite(nodata):
            valid &= dem != nodata
        d = np.where(valid, rdsm, np.nan)
        h = np.where(valid, dem, np.nan)
        fit = calib.fit_affine(d, h)
        if kind == "degenerate":
            assert fit.degenerate is True
            assert fit.method == "degenerate_low_relief"
            assert fit.calibration_warning == "near_zero_scale"
        else:
            assert fit.degenerate is False
            assert "ransac_sign_disagreement" in fit.calibration_warning
            assert (fit.scale < 0.0) == kind.endswith("negative")
        # determinism: a repeated fit on identical inputs is bit-identical
        assert calib.fit_affine(d, h).scale == fit.scale
