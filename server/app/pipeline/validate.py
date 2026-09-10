"""Accuracy evaluation of a produced DSM against reference elevation.

Computes RMSE, MAE, and Pearson correlation over NaN-masked overlap, and
renders a difference heatmap PNG for the validation overlay.
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
    if np.std(p) > 0 and np.std(r) > 0:
        corr = float(np.corrcoef(p, r)[0, 1])
    else:
        corr = 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "correlation": corr,
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
