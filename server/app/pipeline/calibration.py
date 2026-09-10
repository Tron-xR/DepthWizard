"""Scale calibration: convert relative depth (0..1 rDSM) to absolute metric
elevation using a reference DEM (SRTM/Copernicus) or user GCPs.

Method (per 03/05/09 docs): fit r = a * d + b (affine) via ordinary least
squares, then robustified with RANSAC to reject outliers (water, clouds,
buildings, vegetation) before refitting on the inlier set.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Reproducible 80/20 train / held-out split applied in fit_affine: the held-out
# 20% is used purely for validation (RMSE/MAE/correlation) and is NEVER seen by
# the RANSAC+OLS fit. Fixed seed so every run and later /validate calls hold out
# the same pixels.
_TRAIN_FRACTION = 0.8
_SPLIT_SEED = 42


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

    rng = np.random.default_rng(_SPLIT_SEED)
    order = rng.permutation(d.size)
    n_train = int(round(d.size * _TRAIN_FRACTION))
    train, held = order[:n_train], order[n_train:]
    d_train, h_train = d[train], h[train]

    X = d_train.reshape(-1, 1)

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
        )
    inlier_mask = ransac.inlier_mask_

    # Refit OLS on inliers for a stable, low-variance estimate
    lr = LinearRegression().fit(X[inlier_mask], h_train[inlier_mask])
    scale = float(lr.coef_[0])
    offset = float(lr.intercept_)
    n_inliers = int(inlier_mask.sum())
    pred = scale * d_train + offset
    rmse = float(np.sqrt(np.mean((pred - h_train) ** 2)))

    return CalibrationFit(
        scale=scale,
        offset=offset,
        n_samples=int(d.size),
        n_inliers=n_inliers,
        rmse=rmse,
        method="ransac_affine",
        held_relative=d[held],
        held_reference=h[held],
    )


def apply_fit(relative_d: np.ndarray, fit: CalibrationFit) -> np.ndarray:
    """Apply an affine fit to a relative DSM to produce metric elevation."""
    return (relative_d.astype("float32") * fit.scale + fit.offset).astype("float32")


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
