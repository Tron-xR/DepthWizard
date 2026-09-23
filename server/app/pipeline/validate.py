"""Accuracy evaluation of a produced DSM against reference elevation.

Computes RMSE, MAE, and Pearson correlation over NaN-masked overlap, and
renders a difference heatmap PNG for the validation overlay.

Correlation policy (canonical, shared by /evaluate and /validate):
  - r == 0 means "zero LINEAR correlation" - a real, reportable number.
  - An UNDEFINED correlation (either side has zero variance, or fewer than 2
    samples) is a DIFFERENT fact and must never be collapsed into 0.0. It is
    returned as correlation=None with an explicit `correlation_reason`, so a
    failure to measure is never presented as a real measured value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ValidationMetrics:
    rmse: float
    mae: float
    correlation: float
    n: int
    reference_source: str
    landscape_type: str


def correlation_and_reason(predicted: np.ndarray,
                           reference: np.ndarray) -> tuple:
    """Canonical Pearson correlation + explicit reason when UNDEFINED.

    Returns (float, None) when defined, (None, reason_string) when not. r keeps
    its exact sign and magnitude; nothing is forced or flipped. This is the
    ONLY Pearson implementation used by the API, so /evaluate and /validate
    cannot drift apart.
    """
    p = np.asarray(predicted, dtype="float64").ravel()
    r = np.asarray(reference, dtype="float64").ravel()
    keep = np.isfinite(p) & np.isfinite(r)
    p, r = p[keep], r[keep]
    if p.size < 2:
        return None, f"fewer than 2 valid pixels (n={int(p.size)})"
    sp = float(np.std(p))
    sr = float(np.std(r))
    if sp == 0.0 and sr == 0.0:
        return None, "prediction and ground truth are both constant (zero variance)"
    if sp == 0.0:
        return None, "prediction has zero variance (constant)"
    if sr == 0.0:
        return None, "ground truth has zero variance (constant)"
    return float(np.corrcoef(p, r)[0, 1]), None


def compute_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    p = predicted.astype("float64").ravel()
    r = reference.astype("float64").ravel()
    mask = np.isfinite(p) & np.isfinite(r)
    p, r = p[mask], r[mask]
    if p.size < 2:
        raise ValueError("Not enough valid pixels to evaluate")
    diff = p - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    corr, corr_reason = correlation_and_reason(p, r)
    return {
        "rmse": rmse,
        "mae": mae,
        "correlation": corr,
        "correlation_reason": corr_reason,
        "n": int(p.size),
    }


def render_diff_heatmap(predicted: np.ndarray, reference: np.ndarray, path: Path) -> None:
    """Write a PNG color map of (predicted - reference) over the overlap."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    diff = predicted - reference
    absmax = float(np.nanmax(np.abs(diff))) if np.any(np.isfinite(diff)) else 1.0
    if absmax == 0:
        absmax = 1.0
    # -absmax -> blue, 0 -> white, +absmax -> red
    norm = np.clip(diff / absmax, -1.0, 1.0)
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype="float32")
    # r channel: positive
    rgb[:, :, 0] = np.where(norm > 0, norm, 0.0)
    # b channel: negative
    rgb[:, :, 2] = np.where(norm < 0, -norm, 0.0)
    rgb[:, :, 1] = 1.0 - np.abs(norm)  # white center
    rgb = np.clip(rgb, 0, 1)
    img = Image.fromarray((rgb * 255).astype("uint8"), "RGB")
    img.save(path, "PNG")
