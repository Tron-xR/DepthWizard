"""Correlation / calibration policy regression suite (post investigation).

Two facts must stay distinct in every API surface:
  - r == 0 means "zero LINEAR correlation" (a real measured number);
  - UNDEFINED correlation (either side has zero variance, <2 samples) is
    reported as correlation=None with an explicit `correlation_reason`,
    NEVER as a literal 0.0.

Negative-scale / near-constant / near-zero-scale fits are numerically VALID but
flagged as warnings; the fit's exact sign and magnitude are preserved (a
negative scale is never flipped into a fake positive correlation).

Scenarios A-F below were the agreed acceptance spec:
  A  perfect positive/negative linear correlation -> r exact, kept (no flipping)
  B  constant prediction or ground truth          -> correlation=None + reason
  C  both constant                                -> correlation=None + reason
  E  affine transform of one side                 -> r unchanged (scale-invariant)
  F  negative-calibrated synthetic model          -> r stays negative AND fit is
                                                     flagged calibration warning
Plus: near-constant prediction warning, /evaluate surface shape, and the
canonical-helper equality between compute_metrics and evaluate_prediction_truth.
"""
import numpy as np
import pytest

from app.pipeline import calibration as calib
from app.pipeline import evaluation as ev
from app.pipeline import validate as vmod


def _metrics(pred, ref):
    return vmod.compute_metrics(np.asarray(pred, dtype="float64"),
                                np.asarray(ref, dtype="float64"))


# ---- A: perfect positive and negative linear correlation stay exact --------- #
def test_a_perfect_positive_correlation():
    m = _metrics([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert m["correlation"] == pytest.approx(1.0, abs=1e-9)
    assert m["correlation_reason"] is None


def test_a_perfect_negative_correlation_kept_not_flipped():
    m = _metrics([5, 4, 3, 2, 1], [10, 20, 30, 40, 50])
    assert m["correlation"] == pytest.approx(-1.0, abs=1e-9)
    assert m["correlation_reason"] is None


# ---- B: constant side => None + reason, NOT 0.0 ----------------------------- #
def test_b_constant_prediction_is_none_not_zero():
    m = _metrics([3, 3, 3, 3, 3], [10, 20, 30, 40, 50])
    assert m["correlation"] is None
    assert m["correlation_reason"] == "prediction has zero variance (constant)"


def test_b_constant_ground_truth_is_none_not_zero():
    m = _metrics([1, 2, 3, 4, 5], [20, 20, 20, 20, 20])
    assert m["correlation"] is None
    assert m["correlation_reason"] == "ground truth has zero variance (constant)"


# ---- C: both constant -------------------------------------------------------- #
def test_c_both_constant_is_none_with_reason():
    m = _metrics([7, 7, 7, 7, 7], [20, 20, 20, 20, 20])
    assert m["correlation"] is None
    assert "both constant" in m["correlation_reason"]


# ---- E: affine transform leaves correlation unchanged ------------------------ #
def test_e_affine_transform_correlation_invariant():
    m = _metrics([1, 2, 3, 4, 5], [105, 110, 115, 120, 125])
    assert m["correlation"] == pytest.approx(1.0, abs=1e-9)


# ---- F: negative correlation is kept AND the fit is flagged ------------------ #
def test_f_negative_correlation_kept_and_fit_flagged():
    n = 20 * 20
    ramp = np.linspace(0, 1, n).reshape(20, 20).astype("float64")
    dem = (120.0 - 100.0 * ramp).astype("float64")  # d up, h down: inverse model
    fit = calib.fit_affine(ramp, dem)
    assert fit.scale == pytest.approx(-100.0, abs=1e-3)   # kept, NOT abs()'d
    assert fit.degenerate is False
    assert fit.calibration_status == "warning"
    assert fit.calibration_warning == "negative_scale"

    # Raw (uncalibrated) relative-depth correlation stays NEGATIVE - the sign
    # is never flipped anywhere in the API.
    raw = ev.evaluate_prediction_truth(ramp, dem, mode="relative")
    assert raw["correlation"] == pytest.approx(-1.0, abs=1e-9)
    assert raw["correlation_reason"] is None
    assert raw["calibrated"] is False

    # In calibrated mode the surface is FIT to the truth (scale=-100), so its
    # held-out correlation is legitimately +1 - an affine correction, not a
    # sign fake. The negative scale is still surfaced as the warning + value.
    cal = ev.evaluate_prediction_truth(ramp, dem, mode="calibrated")
    assert cal["correlation"] == pytest.approx(1.0, abs=1e-6)
    assert cal["calibration_scale"] == pytest.approx(-100.0, abs=1e-3)
    assert cal["calibration_warning"] == "negative_scale"


# ---- near-constant prediction is a warning on a VALID fit -------------------- #
def test_near_constant_prediction_warns_but_still_fits():
    n = 32 * 32
    ramp = np.linspace(0, 1, n).reshape(32, 32)
    d = 0.5 + 0.001 * ramp        # prediction effectively flat
    h = 50.0 + 100.0 * ramp       # strong true relief
    fit = calib.fit_affine(d, h)
    assert fit.scale > 0
    assert fit.degenerate is False
    assert fit.calibration_status == "warning"
    assert fit.calibration_warning == "prediction_near_constant"


# ---- healthy scene: no warnings, no reason ----------------------------------- #
def test_healthy_scene_status_valid_without_warning():
    n = 32 * 32
    ramp = np.linspace(0, 1, n).reshape(32, 32)
    dem = 5.0 + 10.0 * ramp
    res = ev.evaluate_prediction_truth(ramp, dem, mode="calibrated")
    assert res["correlation"] == pytest.approx(1.0, abs=1e-6)
    assert res["correlation_reason"] is None
    assert res["calibration_status"] == "valid"
    assert res["calibration_warning"] is None
    assert res["calibration_scale"] == pytest.approx(10.0, abs=0.5)
    assert res["degenerate_calibration"] is False
    assert res["mae"] is not None and res["rmse"] is not None


# ---- /evaluate surface: constant prediction (relative mode) ------------------ #
def test_evaluate_constant_pred_is_none_with_reason():
    truth = np.linspace(0, 50, 256).reshape(16, 16)
    res = ev.evaluate_prediction_truth(np.ones((16, 16)), truth, mode="relative")
    assert res["correlation"] is None
    assert res["correlation_reason"] == "prediction has zero variance (constant)"
    assert "no variance" in res["reason"]  # variance guard fires before mode guard
    assert res["calibrated"] is False


# ---- canonical equality: compute_metrics == evaluate path correlation -------- #
def test_evaluate_and_compute_metrics_share_one_formula():
    rng = np.random.default_rng(7)
    pred = rng.uniform(0.2, 0.9, (48, 48))
    truth = 10.0 + 40.0 * rng.uniform(0, 1, (48, 48))
    res = ev.evaluate_prediction_truth(pred, truth, mode="calibrated")
    assert res["correlation"] is not None
    held = calib.fit_affine(pred, truth)
    m = vmod.compute_metrics(held.scale * held.held_relative + held.offset,
                             held.held_reference)
    assert res["correlation"] == pytest.approx(m["correlation"], abs=1e-12)
    assert res["correlation_reason"] is None