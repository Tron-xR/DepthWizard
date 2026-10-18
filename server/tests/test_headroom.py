"""Checks for the HR headroom module (synthetic arrays, no real tiles)."""
import numpy as np
import pytest

from app.pipeline import headroom


def test_arm_reports_known_bias_and_rmse():
    truth = np.arange(100, dtype="float32").reshape(10, 10)
    mask = np.ones(truth.shape, dtype=bool)
    arm = headroom._arm(truth + 2.0, truth, mask)
    assert arm["n"] == 100
    assert arm["bias"] == pytest.approx(2.0)
    assert arm["rmse"] == pytest.approx(2.0)
    assert arm["mae"] == pytest.approx(2.0)
    assert arm["r"] == pytest.approx(1.0)


def test_zero_mean_oracle_leaves_no_bias():
    rng = np.random.default_rng(0)
    truth = rng.normal(500.0, 20.0, size=(40, 40)).astype("float64")
    pred = (0.8 * truth - 30.0).astype("float64")
    mask = np.ones(truth.shape, dtype=bool)
    raw = headroom._arm(pred, truth, mask)
    mu = float(np.mean((pred - truth)[mask]))
    corrected = headroom._arm(pred - mu, truth, mask)
    assert abs(raw["bias"]) > 1.0
    assert corrected["bias"] == pytest.approx(0.0, abs=1e-3)
    assert corrected["rmse"] <= raw["rmse"] + 1e-9


def test_decomposition_identity_matches_var_formula():
    rng = np.random.default_rng(1)
    dtm = rng.normal(1000.0, 50.0, size=(64, 64))
    ndsm = np.abs(rng.normal(5.0, 3.0, size=(64, 64)))
    dsm = dtm + ndsm
    var_dsm = float(np.var(dsm))
    var_dtm = float(np.var(dtm))
    var_ndsm = float(np.var(ndsm))
    cov = float(np.mean((dtm - dtm.mean()) * (ndsm - ndsm.mean())))
    assert var_dsm == pytest.approx(var_dtm + var_ndsm + 2.0 * cov, rel=1e-9)
