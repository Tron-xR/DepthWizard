"""Accuracy evaluation for matched RGB + ground-truth DEM pairs (POST /evaluate).

This is a separate concern from the job-based /validate path, which compares the
produced DSM against an auto-fetched reference DEM (SRTM/Copernicus). Here the
caller uploads (a) an RGB image and (b) its *corresponding ground-truth DEM*,
assumed pixel-aligned (equal dimensions) or georeferenced enough to resample.

The prediction pipeline output is a *relative* normalized depth (0..1), NOT
absolute elevation. MAE/RMSE in absolute units therefore require a scale/offset
callibration. We reuse `calibration.fit_affine`, which fits on 80% of valid
pixels (RANSAC + OLS refit) and holds out the other 20%; the reported
MAE/RMSE/correlation are computed on that held-out 20% only, so the calibration
is never fit and scored on the same pixels. Correlation is scale-invariant and
needs no calibration, but we still report it on the held-out set in calibrated
mode so all three numbers describe the same (out-of-sample) pixels.

Reused (NOT reimplemented here):
  - `validate.compute_metrics` for RMSE/MAE/Pearson on the held-out pairs
  - `calibration.fit_affine` for the robust affine fit + reproducible 80/20 split

Built here as the canonical helpers:
  - `build_valid_mask` (finite + optional raster nodata) - nodata handling did
    not exist anywhere before; other paths mask only NaNs inside compute_metrics.
  - `align_dem_to_prediction` (raster-aware bilinear resample via
    rasterio.warp.reproject, guarded by georeferencing on BOTH inputs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import calibration as calib
from .validate import compute_metrics, correlation_and_reason

# Comparison-image layout (used by render_height_comparison and its tests).
_PANEL_GAP = 4
_LABEL_H = 18

# "terrain" style colormap anchors: (position, RGB) low->high elevation.
_TERRAIN_STOPS = (
    (0.00, (46, 84, 40)),      # dark green
    (0.35, (130, 160, 100)),   # green-tan
    (0.65, (194, 174, 116)),   # tan
    (0.85, (140, 96, 48)),     # brown
    (1.00, (250, 250, 250)),   # white (peaks)
)


def build_valid_mask(predicted: np.ndarray, truth: np.ndarray,
                     *, nodata: Optional[float] = None) -> np.ndarray:
    """Boolean mask of pixel-aligned pairs that may enter the metrics.

    Keeps pixels where BOTH arrays are finite (NaN/inf dropped) and, when the
    DEM declares a raster nodata value, pixels whose elevation differs from it.
    """
    valid = np.isfinite(predicted) & np.isfinite(truth)
    if nodata is not None and np.isfinite(nodata):
        valid = valid & (truth != nodata)
    return valid


def align_dem_to_prediction(predicted: np.ndarray, dem: np.ndarray, *,
                            predicted_transform=None, predicted_crs=None,
                            dem_transform=None, dem_crs=None) -> np.ndarray:
    """Return the DEM resampled onto the prediction grid.

    Equal dimensions: compared directly (documented pixel-aligned pair) - the
    evaluation then assumes pixel-for-pixel correspondence with NO sub-pixel
    shift correction: without CRS there is no ground truth to align against,
    and any blind shift would be guessing. Different dimensions: raster-aware
    bilinear resample, but ONLY when both input carry enough spatial metadata
    (transform + CRS) to know what the grids mean. Without georeferencing we
    refuse to guess spatial correspondence.
    """
    predicted = np.asarray(predicted)
    dem = np.asarray(dem)
    if dem.shape == predicted.shape:
        return dem

    has_geo = (
        predicted_transform is not None and predicted_crs is not None
        and dem_transform is not None and dem_crs is not None
    )
    if not has_geo:
        raise ValueError(
            f"Prediction grid {predicted.shape} differs from ground-truth DEM "
            f"grid {dem.shape} and the pair carries no spatial metadata for "
            "resampling. Provide pixel-aligned matching dimensions, or input "
            "that is georeferenced on both sides."
        )

    from rasterio.warp import Resampling, reproject

    dst = np.empty(predicted.shape, dtype="float32")
    reproject(
        source=dem.astype("float32"),
        destination=dst,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=predicted_transform,
        dst_crs=predicted_crs,
        resampling=Resampling.bilinear,
        # Areas outside the source DEM's extent must become nodata, not a
        # made-up elevation like 0 (which is a legitimate elevation value).
        dst_nodata=np.nan,
    )
    return dst


def evaluate_prediction_truth(relative_depth: np.ndarray, dem: np.ndarray, *,
                              nodata: Optional[float] = None,
                              mode: str = "calibrated") -> dict:
    """Compare the pipeline's relative depth against a pixel-matched DEM.

    predicted/truth must already be aligned on the SAME grid (see
    align_dem_to_prediction); dimensions are validated here.

    mode:
      "calibrated" - fit scale/offset on 80% of valid pixels (via fit_affine),
                     report MAE/RMSE/correlation on the held-out 20% only.
      "relative"   - raw relative-depth comparison: correlation only; MAE/RMSE
                     are returned as null with `reason="relative depth,
                     uncalibrated"` because 0..1 relative depth is not
                     elevation in meters.
    """
    relative_depth = np.asarray(relative_depth, dtype="float64")
    dem = np.asarray(dem, dtype="float64")
    if relative_depth.shape != dem.shape:
        raise ValueError(
            f"Prediction shape {relative_depth.shape} != DEM shape {dem.shape}; "
            "align the pair onto the same grid first."
        )

    total_pixels = int(relative_depth.size)
    valid = build_valid_mask(relative_depth, dem, nodata=nodata)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No valid pixels to evaluate (all NaN/inf/DEM-nodata).")

    pred_v = relative_depth[valid]
    truth_v = dem[valid]

    base = {
        "valid_pixels": n_valid,
        "total_pixels": total_pixels,
        "calibrated": False,
        "held_out_pixel_count": 0,
        "predicted_elevation": None,
        "ground_truth": None,
        "degenerate_calibration": False,
        "calibration_reason": None,
        # New (additive) correlation + calibration health fields, present on
        # EVERY branch so the response shape never varies with the outcome.
        "correlation_reason": None,
        "calibration_status": None,
        "calibration_warning": None,
        "calibration_scale": None,
        "calibration_offset": None,
        "raw_correlation_signed": None,
        "scale_sign": None,
        "polarity_inverted": None,
        "polarity_reason": None,
    }

    # No variance on either side => neither an affine calibration nor a
    # correlation is meaningfully defined.
    if np.std(pred_v) == 0.0 or np.std(truth_v) == 0.0:
        corr, corr_reason = correlation_and_reason(pred_v, truth_v)
        return {
            **base,
            "mae": None, "rmse": None,
            "correlation": corr,
            "correlation_reason": corr_reason,
            "raw_correlation_signed": corr,
            "evaluated_units": "calibrated absolute elevation",
            "reason": "no variance in prediction and/or ground truth: calibration and "
                      "correlation are undefined",
        }

    if mode != "calibrated":
        corr, corr_reason = correlation_and_reason(pred_v, truth_v)
        return {
            **base,
            "mae": None, "rmse": None,
            "correlation": corr,
            "correlation_reason": corr_reason,
            "raw_correlation_signed": corr,
            "evaluated_units": "relative depth (uncalibrated)",
            "reason": "relative depth, uncalibrated",
        }

    if n_valid < 16:
        corr, corr_reason = correlation_and_reason(pred_v, truth_v)
        return {
            **base,
            "mae": None, "rmse": None,
            "correlation": corr,
            "correlation_reason": corr_reason,
            "raw_correlation_signed": corr,
            "evaluated_units": "relative depth (uncalibrated)",
            "reason": "fewer than 16 valid pixels: not enough to fit the scale/offset "
                      "calibration; reporting raw relative-depth correlation only",
        }

    # NaN-mask both grids so fit_affine's own valid-mask + fixed-seed 80/20
    # split sees the same pixel population as our mask.
    pred_masked = np.where(valid, relative_depth, np.nan)
    truth_masked = np.where(valid, dem, np.nan)
    fit = calib.fit_affine(pred_masked, truth_masked)

    if fit.degenerate:
        corr, corr_reason = correlation_and_reason(pred_v, truth_v)
        return {
            **base,
            "mae": None, "rmse": None,
            "correlation": corr,
            "correlation_reason": corr_reason,
            "raw_correlation_signed": corr,
            "scale_sign": "+" if fit.scale >= 0 else "-",
            "polarity_inverted": fit.polarity_inverted,
            "polarity_reason": fit.polarity_reason,
            "evaluated_units": "calibrated absolute elevation",
            "reason": "calibration degenerate: the fitted (80%) surface had no variance "
                      "(e.g. flat prediction), so no scale/offset is meaningful; "
                      "reporting raw relative-depth correlation only",
            # New (additive) machine-readable flag + human reason for the flag.
            "degenerate_calibration": True,
            "calibration_reason": calib.degenerate_reason(fit),
            "calibration_status": fit.calibration_status,
            "calibration_warning": fit.calibration_warning,
            "calibration_scale": fit.scale,
            "calibration_offset": fit.offset,
        }

    # Score ONLY the held-out 20% - the calibration was fit on the other 80%.
    held_pred = fit.scale * fit.held_relative + fit.offset
    held_truth = fit.held_reference

    metrics = compute_metrics(held_pred, held_truth)
    corr, corr_reason = correlation_and_reason(held_pred, held_truth)

    # Uncalibrated model polarity on the SAME held-out 20%: correlation of the
    # raw relative depth vs the reference, before any calibration sign flip.
    raw_held_corr, _ = correlation_and_reason(fit.held_relative, fit.held_reference)
    # A negative fitted scale means the model output is inverted relative to the
    # reference (calibrated correlation is polarity-flipped); say so explicitly
    # instead of reporting a positive correlation that hides the inversion.
    if fit.scale < 0 and corr is not None:
        corr_reason = (
            "negative calibration scale: model output is inverted relative to the "
            f"reference (raw_correlation_signed={raw_held_corr:+.3f}); the reported "
            "calibrated correlation has the flipped polarity - use "
            "raw_correlation_signed for the unflipped model signal"
        )

    # Calibrated elevation over the whole grid (NaN outside the valid mask),
    # exposed so the route can render a visualization the caller opted into.
    predicted_elevation = np.where(valid, fit.scale * relative_depth + fit.offset, np.nan)
    ground_truth = np.where(valid, dem, np.nan)

    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "correlation": corr,
        "correlation_reason": corr_reason,
        "valid_pixels": n_valid,
        "total_pixels": total_pixels,
        "held_out_pixel_count": int(fit.held_relative.size),
        "calibrated": True,
        "evaluated_units": "calibrated absolute elevation (scale/offset fit on 80% of "
                           "valid pixels, scored on the held-out 20%)",
        "reason": None,
        "degenerate_calibration": False,
        "calibration_reason": None,
        "calibration_status": fit.calibration_status,
        "calibration_warning": fit.calibration_warning,
        "calibration_scale": fit.scale,
        "calibration_offset": fit.offset,
        "raw_correlation_signed": raw_held_corr,
        "scale_sign": "+" if fit.scale >= 0 else "-",
        "polarity_inverted": fit.polarity_inverted,
        "polarity_reason": fit.polarity_reason,
        "predicted_elevation": predicted_elevation,
        "ground_truth": ground_truth,
    }


def _terrain_lut(n: int = 256) -> np.ndarray:
    """Terrain-style (n,3) LUT interpolated from _TERRAIN_STOPS."""
    pos = np.linspace(0.0, 1.0, n)
    xs = tuple(s[0] for s in _TERRAIN_STOPS)
    return np.stack([
        np.interp(pos, xs, tuple(s[1][c] for s in _TERRAIN_STOPS))
        for c in range(3)
    ], axis=1)


def _elevation_to_rgb(arr: np.ndarray, vmin: float, vmax: float,
                      lut: np.ndarray) -> np.ndarray:
    """Map elevation to LUT colors under a SHARED (vmin, vmax) normalization.

    NaN pixels become neutral grey (they are nodata, not low elevation).
    """
    span = vmax - vmin if vmax > vmin else 1.0
    norm = np.clip((arr - vmin) / span, 0.0, 1.0)
    idx = (norm * (lut.shape[0] - 1)).astype("int32")
    grey = np.array([128, 128, 128], dtype="uint8")
    valid = np.isfinite(arr)
    h, w = arr.shape
    rgb = np.where(valid[:, :, None], lut[idx], grey[None, None, :])
    return rgb.astype("uint8")


def render_height_comparison(rgb: np.ndarray, truth: np.ndarray,
                             predicted: np.ndarray, path: Path) -> None:
    """Save a 3-panel comparison PNG: input RGB | ground truth | predicted.

    Ground-truth and predicted panels share ONE (vmin, vmax) computed over all
    valid pixels of both arrays, so the two surfaces are visually comparable
    (a shared color means the same elevation). NaN == nodata -> grey.
    """
    from PIL import Image, ImageDraw

    rgb = np.asarray(rgb)
    if rgb.max() <= 1.05:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype("uint8")
    truth = np.asarray(truth, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")

    finite = np.concatenate([truth.ravel(), predicted.ravel()])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    vmin, vmax = float(finite.min()), float(finite.max())

    lut = _terrain_lut()
    h, w = truth.shape
    tall = max(h, rgb.shape[0], predicted.shape[0])
    panels = [
        rgb,
        _elevation_to_rgb(truth, vmin, vmax, lut),
        _elevation_to_rgb(predicted, vmin, vmax, lut),
    ]
    out_w = 3 * w + 2 * _PANEL_GAP
    out_h = tall + _LABEL_H
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(zip(("input RGB", "ground truth", "predicted (calibrated)"),
                                           panels)):
        x = i * (w + _PANEL_GAP)
        if panel.shape[0] != tall:
            panel = np.repeat(panel[:, :, None], 1, axis=2)[:, :, 0] if panel.ndim == 2 else panel
        if panel.shape[1] != w:
            panel = np.repeat(panel, max(1, w // panel.shape[1]), axis=1)[:, :w]
        canvas.paste(Image.fromarray(panel, "RGB"), (x, _LABEL_H))
        draw.text((x + 4, 3), label, fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG")