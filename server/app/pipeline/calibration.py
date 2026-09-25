"""Scale calibration: convert relative depth (0..1 rDSM) to absolute metric
elevation using a reference DEM (SRTM/Copernicus) or user GCPs.

Method (per 03/05/09 docs): fit r = a * d + b (affine) via ordinary least
squares, then robustified with RANSAC to reject outliers (water, clouds,
buildings, vegetation) before refitting on the inlier set.

Negative-scale policy: all four depth backends (pix2pix GAN, IMELE, finetuned
decoder, Depth-Anything) normalize output so brighter = higher elevation, so a
POSITIVE scale is the project's expected correlation direction between predicted
relative depth and true elevation. A negative fitted scale contradicts that
convention (it almost always means RANSAC locked onto a near-orthogonal /
near-constant prediction). It is FLAGGED as a warning and NEVER flipped: abs()
would fabricate a positive association the model does not have.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

from .. import config

# Reproducible 80/20 train / held-out split applied in fit_affine: the held-out
# 20% is used purely for validation (RMSE/MAE/correlation) and is NEVER seen by
# the RANSAC+OLS fit. Fixed seed so every run and later /validate calls hold out
# the same pixels.
_TRAIN_FRACTION = 0.8
_SPLIT_SEED = 42

# Near-degenerate guard: a fit whose predicted elevation spans less than this
# fraction of the training terrain's own elevation range (np.ptp(h_train)) is
# effectively flat, so |scale| is meaningless. Relative, not absolute, so a
# genuinely low-relief scene (e.g. a lake that really is 0-5 m) scales its
# guard down with it instead of being flagged for being small.
_LOW_RELIEF_FRACTION = 0.02

# Warning kinds attached to an otherwise-valid calibration fit. They are
# informational (never null metrics or gate completion).
_WARN_NEGATIVE_SCALE = "negative_scale"
_WARN_NEAR_CONSTANT = "prediction_near_constant"
_WARN_NEAR_ZERO_SCALE = "near_zero_scale"
_WARN_INSUFFICIENT_VARIANCE = "insufficient_variance"
_WARN_RANSAC_SIGN_DISAGREEMENT = "ransac_sign_disagreement"

# Model-polarity threshold on the TRAIN-split raw correlation (relative depth
# vs true elevation): below this the model output is treated as inverted
# relative to the reference (brighter=lower). Independent of calibration
# stability; a RANSAC sign flip does NOT imply inversion and vice versa.
_POLARITY_NEG_THRESHOLD = -0.25


@dataclass
class CalibrationFit:
    scale: float
    offset: float
    n_samples: int
    n_inliers: int
    rmse: float
    method: str
    # held-out (validation-only) samples: (relative depth, true elevation).
    # Calibrated absolute DSM is evaluated on these pixels, never fit on them.
    held_relative: Optional[np.ndarray] = None
    held_reference: Optional[np.ndarray] = None
    # True when the training 80% is degenerate (constant or near-constant), so
    # no meaningful scale/offset exists; scale=0/offset=median is returned.
    degenerate: bool = False
    # Calibration health (additive, informational): a numerically VALID fit can
    # still be semantically suspicious - near-constant prediction, near-zero
    # scale, or a NEGATIVE scale (see _WARN_*). Status is "valid" | "warning" |
    # "degenerate"; warnings never null the metrics and never alter scale/offset.
    calibration_status: str = "valid"
    calibration_warning: Optional[str] = None
    pred_std: Optional[float] = None   # std of prediction used for the fit
    ref_std: Optional[float] = None    # std of reference used for the fit
    # Model polarity, measured on the TRAIN-split raw correlation (independent
    # of calibration): True when it is clearly negative (model output inverted
    # relative to the reference), False when positive, None when undefined
    # (constant prediction/reference on the train split).
    polarity_inverted: Optional[bool] = None
    polarity_reason: Optional[str] = None


def fit_affine(relative_d: np.ndarray, reference_h: np.ndarray) -> CalibrationFit:
    """Fit elevation = scale * d + offset using matching pixels.

    NaN/NaN mask is dropped. A reproducible 80/20 split reserves 20% of sample
    points purely for validation; the fit (RANSAC then OLS refit on inliers)
    runs on the training 80% only, so the held-out set is never seen during
    calibration. RANSAC robustifies against outlier elevation (buildings,
    vegetation) within the training set.

    relative_d, reference_h: same-shaped 2D arrays.
    """
    d = relative_d.ravel().astype("float64")
    h = reference_h.ravel().astype("float64")
    mask = np.isfinite(d) & np.isfinite(h)
    d, h = d[mask], h[mask]
    if d.size < 16:
        raise ValueError("Not enough valid sample pairs for calibration")

    sp = float(np.std(d))
    sr = float(np.std(h))

    rng = np.random.default_rng(_SPLIT_SEED)
    order = rng.permutation(d.size)
    n_train = int(round(d.size * _TRAIN_FRACTION))
    train, held = order[:n_train], order[n_train:]
    d_train, h_train = d[train], h[train]

    X = d_train.reshape(-1, 1)

    # Model polarity from the TRAIN-split raw correlation (never the full grid,
    # never held-out). Independent of whatever calibration sign RANSAC later
    # picks: a RANSAC flip is a FIT artifact, not evidence about the model.
    if np.std(d_train) == 0.0 or np.std(h_train) == 0.0:
        polarity_inverted, polarity_reason = None, (
            "undefined: prediction or reference is constant on the training split")
    else:
        _train_corr = float(np.corrcoef(d_train, h_train)[0, 1])
        polarity_inverted = bool(_train_corr < _POLARITY_NEG_THRESHOLD)
        polarity_reason = (
            f"train-split raw correlation {_train_corr:+.3f} < {_POLARITY_NEG_THRESHOLD}: "
            "model output is inverted relative to the reference"
            if polarity_inverted else
            f"train-split raw correlation {_train_corr:+.3f}: "
            "model polarity agrees with the reference"
        )

    # Degenerate fit: the training 80% holds (nearly) no variance (e.g. a
    # flat GAN output), so RANSAC would raise or fit garbage. Return a constant
    # elevation fit instead of crashing (happens on flat imagery / constant rDSM).
    if np.ptp(d_train) == 0.0:
        offset = float(np.median(h_train))
        return CalibrationFit(
            scale=0.0, offset=offset,
            n_samples=int(d.size), n_inliers=0,
            rmse=float(np.sqrt(np.mean((h_train - offset) ** 2))),
            method="degenerate_constant", degenerate=True,
            held_relative=d[held], held_reference=h[held],
            calibration_status="degenerate",
            calibration_warning=_WARN_NEAR_CONSTANT,
            pred_std=sp, ref_std=sr,
            polarity_inverted=polarity_inverted,
            polarity_reason=polarity_reason,
        )

    # RANSAC to find inliers on the training data
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=8,
        residual_threshold=None,
        random_state=0,
    )
    try:
        ransac.fit(X, h_train)
    except ValueError:
        # RANSAC can still bail when the training sample is numerically
        # degenerate; degrade to a plain constant fit rather than 500-ing.
        offset = float(np.mean(h_train))
        return CalibrationFit(
            scale=0.0, offset=offset,
            n_samples=int(d.size), n_inliers=0,
            rmse=float(np.sqrt(np.mean((h_train - offset) ** 2))),
            method="degenerate_constant", degenerate=True,
            held_relative=d[held], held_reference=h[held],
            calibration_status="degenerate",
            calibration_warning=_WARN_INSUFFICIENT_VARIANCE,
            pred_std=sp, ref_std=sr,
            polarity_inverted=polarity_inverted,
            polarity_reason=polarity_reason,
        )
    inlier_mask = ransac.inlier_mask_

    # Refit OLS on the RANSAC inliers (stable, low-variance estimate) AND on the
    # SAME train split as a whole. When the two disagree on sign, RANSAC has
    # locked onto a near-orthogonal sub-population (a documented artifact on
    # noisy predictors, e.g. sparse_01: raw corr +0.585 but RANSAC scale -159).
    # Then trust the OLS sign of the full train split and flag the flip.
    ransac_lr = LinearRegression().fit(X[inlier_mask], h_train[inlier_mask])
    ols_lr = LinearRegression().fit(X, h_train)
    ransac_scale = float(ransac_lr.coef_[0])
    ols_scale = float(ols_lr.coef_[0])
    ols_offset = float(ols_lr.intercept_)
    n_inliers = int(inlier_mask.sum())

    def _sign(v):
        return 1 if v >= 0.0 else -1

    if _sign(ransac_scale) != _sign(ols_scale):
        scale, offset = ols_scale, ols_offset
        sign_disagreement = True
    else:
        scale, offset = ransac_scale, float(ransac_lr.intercept_)
        sign_disagreement = False

    pred = scale * d_train + offset
    rmse = float(np.sqrt(np.mean((pred - h_train) ** 2)))

    # Near-degenerate guard: the fitted scale maps the full relative-depth range
    # to a span smaller than _LOW_RELIEF_FRACTION * the training terrain's own
    # elevation range, so the calibrated surface is effectively flat. Typical on
    # low-relief tiles (lakes, plains) where RANSAC locks onto the flat majority
    # and |scale| collapses to ~0 even when np.ptp(d_train) != 0. Flag it and
    # return a constant elevation (flat result) instead of a bogus near-zero
    # scale that exports an all-black heightmap. The job still completes.
    relief = float(np.ptp(h_train))
    pred_span = abs(scale) * float(np.ptp(d_train))
    if relief > 0.0 and pred_span < _LOW_RELIEF_FRACTION * relief:
        offset = float(np.median(h_train))
        return CalibrationFit(
            scale=0.0, offset=offset,
            n_samples=int(d.size), n_inliers=n_inliers,
            rmse=float(np.sqrt(np.mean((h_train - offset) ** 2))),
            method="degenerate_low_relief", degenerate=True,
            held_relative=d[held], held_reference=h[held],
            calibration_status="degenerate",
            calibration_warning=_WARN_NEAR_ZERO_SCALE,
            pred_std=sp, ref_std=sr,
            polarity_inverted=polarity_inverted,
            polarity_reason=polarity_reason,
        )

    # Classify fit health (informational). The fit is numerically VALID and
    # returned with its exact sign/magnitude intact - only a warning is set:
    #   negative_scale        -> priority 0: contradicts the brighter=higher
    #                             convention; never abs()'d (would fake a positive
    #                             association the model does not have).
    #   prediction_near_constant -> prediction variance is below
    #                             config.MIN_RELATIVE_STD of the reference's
    #                             own variance (model output effectively flat).
    rel_std = sp / sr if sr > 0.0 else float("inf")
    warning = None
    if sign_disagreement:
        warning = _WARN_RANSAC_SIGN_DISAGREEMENT
    elif scale < 0.0:
        warning = _WARN_NEGATIVE_SCALE
    elif rel_std < config.MIN_RELATIVE_STD:
        warning = _WARN_NEAR_CONSTANT
    status = "warning" if warning else "valid"

    return CalibrationFit(
        scale=scale,
        offset=offset,
        n_samples=int(d.size),
        n_inliers=n_inliers,
        rmse=rmse,
        method="ransac_affine",
        held_relative=d[held],
        held_reference=h[held],
        calibration_status=status,
        calibration_warning=warning,
        pred_std=sp,
        ref_std=sr,
        polarity_inverted=polarity_inverted,
        polarity_reason=polarity_reason,
    )


def apply_fit(relative_d: np.ndarray, fit: CalibrationFit) -> np.ndarray:
    """Apply an affine fit to a relative DSM to produce metric elevation."""
    return (relative_d.astype("float32") * fit.scale + fit.offset).astype("float32")


def degenerate_reason(fit: CalibrationFit) -> Optional[str]:
    """Human-readable reason for a flagged (degenerate) calibration fit, or
    None when the fit is normal. Consumed by the response schemas."""
    if not fit.degenerate:
        return None
    if fit.method == "degenerate_low_relief":
        return ("low relief detected in this tile; calibration scale near zero, "
                "elevation treated as flat")
    return ("no variance in the predicted (fitted) surface; calibration scale "
            "is zero, elevation treated as flat")


def save_fit(fit: CalibrationFit, path: Path) -> None:
    """Persist a fit (incl. its held-out validation-only set) as a .npz.

    /validate re-scores the SAME 20% of pixels fit_affine reserved, from disk,
    instead of comparing every pixel (which would leak the fit's training 80%).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    held_r = fit.held_relative.astype("float64") if fit.held_relative is not None else np.empty(0)
    held_h = fit.held_reference.astype("float64") if fit.held_reference is not None else np.empty(0)
    np.savez(
        path,
        scale=np.float64(fit.scale),
        offset=np.float64(fit.offset),
        n_samples=np.int64(fit.n_samples),
        n_inliers=np.int64(fit.n_inliers),
        rmse=np.float64(fit.rmse),
        method=np.asarray(fit.method),
        degenerate=np.int8(fit.degenerate),
        calibration_status=np.asarray(fit.calibration_status),
        calibration_warning=np.asarray(fit.calibration_warning or ""),
        polarity_inverted=np.int8(-1) if fit.polarity_inverted is None
        else np.int8(int(fit.polarity_inverted)),
        polarity_reason=np.asarray(fit.polarity_reason or ""),
        held_relative=held_r,
        held_reference=held_h,
    )


def load_fit(path: Path) -> Optional[CalibrationFit]:
    """Reload a fit written by save_fit; None when the file is missing (a
    job processed before holdout persistence existed), so callers can fall back
    to the legacy whole-grid comparison instead of failing."""
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as z:
        held_r = z["held_relative"]
        if held_r.size == 0:
            held_r = held_reference = None
        else:
            held_reference = z["held_reference"]
        # Health fields are additive; older .npz files load as a plain valid fit.
        cal_status = (str(z["calibration_status"].item())
                      if "calibration_status" in z.files else "valid")
        cal_warn_raw = z["calibration_warning"] if "calibration_warning" in z.files else np.asarray("")
        cal_warn_raw = cal_warn_raw.item() if cal_warn_raw.size else ""
        cal_warn = str(cal_warn_raw) if cal_warn_raw else None
        if "polarity_inverted" in z.files:
            p_inv_raw = int(np.asarray(z["polarity_inverted"]).item())
            polarity_inverted = None if p_inv_raw < 0 else bool(p_inv_raw)
        else:
            polarity_inverted = None
        p_reason_raw = z["polarity_reason"] if "polarity_reason" in z.files else np.asarray("")
        p_reason_raw = p_reason_raw.item() if p_reason_raw.size else ""
        polarity_reason = str(p_reason_raw) if p_reason_raw else None
        return CalibrationFit(
            scale=float(z["scale"]),
            offset=float(z["offset"]),
            n_samples=int(z["n_samples"]),
            n_inliers=int(z["n_inliers"]),
            rmse=float(z["rmse"]),
            method=str(z["method"].item()),
            degenerate=bool(int(z["degenerate"])),
            held_relative=held_r,
            held_reference=held_reference,
            calibration_status=cal_status,
            calibration_warning=cal_warn,
            polarity_inverted=polarity_inverted,
            polarity_reason=polarity_reason,
        )


def gcp_affine(gcp_d: np.ndarray, gcp_h: np.ndarray) -> CalibrationFit:
    """Fit affine using explicit ground-control-point (relative depth, true height) pairs."""
    X = np.asarray(gcp_d, dtype="float64").reshape(-1, 1)
    h = np.asarray(gcp_h, dtype="float64").reshape(-1)
    if X.shape[0] < 2:
        raise ValueError("At least 2 GCPs are required")
    lr = LinearRegression().fit(X, h)
    pred = lr.predict(X)
    rmse = float(np.sqrt(np.mean((pred - h) ** 2)))
    return CalibrationFit(
        scale=float(lr.coef_[0]),
        offset=float(lr.intercept_),
        n_samples=int(X.shape[0]),
        n_inliers=int(X.shape[0]),
        rmse=rmse,
        method="gcp_affine",
    )


def resample_to_grid(src: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resample a 2D array (e.g. reference DEM) to a target (H, W) grid via
    bilinear resampling through Pillow."""
    from PIL import Image

    th, tw = target_shape
    sh, sw = src.shape
    if (sh, sw) == (th, tw):
        return src.astype("float32").copy()

    lo, hi = float(np.nanmin(src)), float(np.nanmax(src))
    norm = np.nan_to_num(src, nan=0.0)
    if hi - lo == 0:
        norm = np.zeros_like(norm)
    else:
        norm = (norm - lo) / (hi - lo)
    img = Image.fromarray((norm * 255).astype("uint8"))
    img = img.resize((tw, th), Image.BILINEAR)
    res = np.asarray(img, dtype="float32") / 255.0 * (hi - lo) + lo
    return res
