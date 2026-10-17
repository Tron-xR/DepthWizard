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

import csv
import hashlib
import json
import sys
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

    # Non-georeferenced evaluation: report the taxonomy status directly even on
    # the early-return branches (constant-prediction etc.) below.
    relative_mode = mode != "calibrated"

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
        "estimate_status": "relative" if relative_mode else None,
        "calibration_inlier_count": None,
        "bias": None,
        "prediction_std": None,
        "reference_std": None,
        "raw_correlation_signed": None,
        "scale_sign": None,
        "polarity_inverted": None,
        "polarity_reason": None,
        # Pre-decision slopes of the scheme-selection step (additive, see
        # calibration.CalibrationFit.ransac_scale / ols_scale).
        "calibration_ransac_scale": None,
        "calibration_ols_scale": None,
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
            "estimate_status": fit.estimate_status,
            "calibration_inlier_count": fit.n_inliers,
            "calibration_ransac_scale": fit.ransac_scale,
            "calibration_ols_scale": fit.ols_scale,
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
        "estimate_status": fit.estimate_status,
        "calibration_inlier_count": fit.n_inliers,
        "bias": float(np.mean(held_pred - held_truth)),
        "prediction_std": float(np.std(held_pred)),
        "reference_std": float(np.std(held_truth)),
        "calibration_ransac_scale": fit.ransac_scale,
        "calibration_ols_scale": fit.ols_scale,
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


# ---------------------------------------------------------------------------
# Phase 1 benchmark harness: manifest -> arms -> per-tile/site/stratum stats.
#
# One CLI:  python -m app.pipeline.evaluation benchmark \
#               --manifest data/manifest.csv --split test --out results/
#
# Protocol (thresholds live in config/acceptance.yaml, frozen up front):
#   - splits are assigned per SITE (region), never per tile; the manifest
#     loader refuses a site that appears in two splits and warns when tiles of
#     different splits sit within --guard-km of each other;
#   - the affine calibration is fit ONLY on a checkerboard of contiguous
#     spatial blocks and scored ONLY on the complement eroded by a guard band,
#     so no evaluated pixel helped fit the scale/offset;
#   - a random-pixel split is reported alongside as a within-scene diagnostic
#     (optimistic by construction; never the headline number);
#   - all arms share ONE valid-pixel mask per tile (intersection across the
#     available arms) so their errors are paired and comparable;
#   - raw (pre-calibration) correlations keep an explicit reason when they are
#     undefined - never coerced to 0; failed/low_confidence tiles stay in every
#     count;
#   - aggregation is site-weighted (mean over site means) and bootstrap CIs
#     resample SITES, not pixels.
# ---------------------------------------------------------------------------

_MANIFEST_REQUIRED_COLUMNS = (
    "tile_id", "site_id", "region", "split", "sensor", "gsd", "crs",
    "reference_product", "reference_type", "reference_res_m",
    "vertical_datum", "stratum",
)
_MANIFEST_OPTIONAL_COLUMNS = (
    "capture_date", "reference_hr_path", "notes", "bounds_wgs84",
)
_MANIFEST_COLUMNS = _MANIFEST_REQUIRED_COLUMNS + _MANIFEST_OPTIONAL_COLUMNS
_BENCHMARK_SPLITS = ("train", "val", "test")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PREDICTION_RUNS = {
    "pix2pix": "benchmark_runs/c3a2841_pix2pix_20261016T134335Z",
    "finetuned": "benchmark_runs/c3a2841_finetuned_20261016T134947Z",
    "imele": "benchmark_runs/c3a2841_imele_20261016T155949Z",
    "da2": "benchmark_runs/c3a2841_da2_20261016T162855Z",
}
# Below this bicubic-baseline RMSE (m) the baseline is numerically
# self-referential (the reference scored against itself via the identity
# reprojection): a fractional "improvement vs bicubic" would divide by float
# noise. Real interpolation error from a genuinely coarser product is >= 0.1 m.
_BICUBIC_SELF_REFERENTIAL_M = 0.01


class ManifestError(ValueError):
    """The manifest cannot be used as-is (structural or leakage problem)."""


def load_manifest(path, *, guard_km: float = 10.0) -> tuple:
    """Read + validate the benchmark manifest. Returns (rows, validation).

    Raises ManifestError when the manifest is structurally unusable or a SITE
    appears in more than one split (tile-level leakage by construction).
    Cross-split tiles closer than guard_km come back as warnings: they taint
    the provenance, so the report records them instead of hiding them.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = []
        for raw in reader:
            row = {k: (v or "").strip() for k, v in raw.items() if k is not None}
            if any(row.values()):
                rows.append(row)
    missing = [c for c in _MANIFEST_REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ManifestError(f"manifest {path} is missing columns: {missing}")
    if not rows:
        raise ManifestError(f"manifest {path} has no tile rows")
    validation = validate_manifest(rows, guard_km=guard_km)
    if validation["errors"]:
        raise ManifestError(
            f"manifest {path} failed validation:\n  "
            + "\n  ".join(validation["errors"]))
    return rows, validation


def validate_manifest(rows, *, guard_km: float = 10.0) -> dict:
    """Structural + leakage validation of already-parsed manifest rows."""
    errors: list = []
    warnings: list = []
    seen: dict = {}
    site_splits: dict = {}
    parsed_bounds: dict = {}
    for i, row in enumerate(rows, start=2):
        where = f"line {i} ({row.get('tile_id') or '?'})"
        for col in _MANIFEST_REQUIRED_COLUMNS:
            if not row.get(col):
                errors.append(f"{where}: missing required column '{col}'")
        tile_id = row.get("tile_id")
        if tile_id:
            if tile_id in seen:
                errors.append(
                    f"{where}: duplicate tile_id (already on line {seen[tile_id]})")
            seen[tile_id] = i
        split = row.get("split")
        if split and split not in _BENCHMARK_SPLITS:
            errors.append(f"{where}: split '{split}' not in {_BENCHMARK_SPLITS}")
        elif split and row.get("site_id"):
            site_splits.setdefault(row["site_id"], set()).add(split)
        for col in ("gsd", "reference_res_m"):
            try:
                if float(row.get(col) or "") <= 0:
                    raise ValueError
            except ValueError:
                errors.append(
                    f"{where}: {col} must be a positive number, got {row.get(col)!r}")
        bounds = row.get("bounds_wgs84")
        if bounds:
            try:
                parsed = tuple(float(v) for v in bounds.split(","))
                if len(parsed) != 4 or parsed[2] <= parsed[0] or parsed[3] <= parsed[1]:
                    raise ValueError
                parsed_bounds[tile_id] = parsed
            except ValueError:
                errors.append(
                    f"{where}: bounds_wgs84 must be 'minx,miny,maxx,maxy'")
    for site, splits in sorted(site_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"site '{site}' appears in multiple splits {sorted(splits)}: "
                "splits are assigned per site, never per tile")
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            ra, rb = rows[a], rows[b]
            if ra.get("split") == rb.get("split"):
                continue
            ba = parsed_bounds.get(ra.get("tile_id"))
            bb = parsed_bounds.get(rb.get("tile_id"))
            if not ba or not bb:
                continue
            gap = _bounds_gap_km(ba, bb)
            if gap < guard_km:
                warnings.append(
                    f"tiles {ra.get('tile_id')} ({ra.get('split')}) and "
                    f"{rb.get('tile_id')} ({rb.get('split')}) are ~{gap:.1f} km "
                    f"apart (< {guard_km:g} km guard): cross-split neighbourhood")
    return {"errors": errors, "warnings": warnings}


def _bounds_gap_km(a, b) -> float:
    """Edge-to-edge gap (km) between two lon/lat rectangles, local planar."""
    lat0 = (a[1] + a[3] + b[1] + b[3]) / 4.0
    kx = 111.32 * np.cos(np.radians(lat0))
    ky = 110.57
    ax0, ay0, ax1, ay1 = a[0] * kx, a[1] * ky, a[2] * kx, a[3] * ky
    bx0, by0, bx1, by1 = b[0] * kx, b[1] * ky, b[2] * kx, b[3] * ky
    dx = max(0.0, ax0 - bx1, bx0 - ax1)
    dy = max(0.0, ay0 - by1, by0 - ay1)
    return float(np.hypot(dx, dy))


def spatial_block_split(shape, *, block: int = 180, guard: int = 8) -> tuple:
    """Checkerboard spatial-block split -> (calibration_mask, eval_mask).

    Deterministic (no seed): blocks with (bi + bj) even are calibration, the
    rest are evaluation, eroded by `guard` pixels so no evaluated pixel sits
    within the guard band of a calibration pixel. Erosion treats the image
    border as evaluable (border_value=1): only proximity to an actual
    calibration block removes a pixel.
    """
    from scipy import ndimage

    h, w = shape
    bi = np.arange(h) // max(1, block)
    bj = np.arange(w) // max(1, block)
    calib_mask = ((bi[:, None] + bj[None, :]) % 2) == 0
    eval_mask = ~calib_mask
    if guard > 0:
        col = np.ones((1, 2 * guard + 1), dtype=bool)
        row = np.ones((2 * guard + 1, 1), dtype=bool)
        eval_mask = ndimage.binary_erosion(eval_mask, structure=col, border_value=1)
        eval_mask = ndimage.binary_erosion(eval_mask, structure=row, border_value=1)
    return calib_mask, eval_mask


def _resample_reference(dem, shape, *, georef=None, resampling: str = "cubic"):
    """Coarse reference DEM on the evaluation grid (real reprojection path).

    georef: (src_transform, src_crs, dst_transform, dst_crs) or None. Without
    georeferencing the stored grids are ASSUMED pixel-aligned (the documented
    fallback of align_dem_to_prediction); differing shapes then refuse to
    guess. On the current manifest the reference and prediction share one
    grid, so the reprojection is the identity - the path stays real for
    reference products that are genuinely coarser.
    """
    if georef and all(part is not None for part in georef):
        src_transform, src_crs, dst_transform, dst_crs = georef
        from rasterio.warp import Resampling, reproject

        method = {"cubic": Resampling.cubic, "nearest": Resampling.nearest}[resampling]
        dst = np.empty(shape, dtype="float32")
        reproject(
            source=dem.astype("float32"), destination=dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=method, dst_nodata=np.nan,
        )
        return dst
    if dem.shape == shape:
        return np.asarray(dem, dtype="float32")
    raise ValueError(
        f"reference grid {dem.shape} != evaluation grid {shape} and no "
        "georeferencing was provided to resample it")


def _arm_stats(pred, truth, mask) -> dict:
    """MAE/RMSE/bias/Pearson of one arm over one pixel mask (NaN-safe)."""
    n = int(np.count_nonzero(mask))
    if n < 2:
        return {"n": n, "mae": None, "rmse": None, "bias": None,
                "correlation": None,
                "correlation_reason": f"fewer than 2 valid pixels (n={n})"}
    metrics = compute_metrics(pred[mask], truth[mask])
    diff = pred[mask].astype("float64") - truth[mask].astype("float64")
    return {"n": metrics["n"], "mae": metrics["mae"], "rmse": metrics["rmse"],
            "bias": float(np.mean(diff)),
            "correlation": metrics["correlation"],
            "correlation_reason": metrics["correlation_reason"]}


def _cap_mask(mask, cap: int, rng) -> np.ndarray:
    """Deterministically subsample a boolean mask to at most `cap` pixels."""
    total = int(np.count_nonzero(mask))
    if total <= cap:
        return mask
    idx = np.flatnonzero(mask.ravel())
    picked = rng.choice(idx, size=cap, replace=False)
    out = np.zeros(mask.shape, dtype=bool)
    out.ravel()[picked] = True
    return out


def _fit_on_mask(pred, dem, mask) -> tuple:
    """calibration.fit_affine restricted to one pixel mask. (fit, error)."""
    n = int(np.count_nonzero(mask))
    if n < 16:
        return None, f"fewer than 16 calibration pixels (n={n})"
    try:
        return calib.fit_affine(np.where(mask, pred, np.nan),
                                np.where(mask, dem, np.nan)), None
    except ValueError as exc:
        return None, f"calibration fit failed: {exc}"


def evaluate_tile_arrays(dem, predictions: dict, *, georef=None,
                         block: int = 180, guard: int = 8,
                         calib_cap: int = 2_000_000, seed: int = 42) -> dict:
    """All arms of one tile on one shared valid-pixel mask.

    predictions: {backend: float array expected on the reference grid}.
    Backends whose grid does not match are skipped and recorded, never
    silently broadcast. `georef` (src+stp transforms/CRS) drives the real
    reference reprojection for the nearest/bicubic baselines.
    """
    dem = np.asarray(dem, dtype="float32")
    shape = dem.shape
    rng = np.random.default_rng(seed)

    preds: dict = {}
    skipped: dict = {}
    for name, pred in predictions.items():
        arr = np.asarray(pred, dtype="float32")
        if arr.shape != shape:
            skipped[name] = f"prediction grid {arr.shape} != reference grid {shape}"
            continue
        preds[name] = arr

    ref_nearest = _resample_reference(dem, shape, georef=georef, resampling="nearest")
    ref_cubic = _resample_reference(dem, shape, georef=georef, resampling="cubic")

    valid = np.isfinite(dem) & np.isfinite(ref_nearest) & np.isfinite(ref_cubic)
    for arr in preds.values():
        valid = valid & np.isfinite(arr)

    calib_mask, eval_mask = spatial_block_split(shape, block=block, guard=guard)
    calib_px = calib_mask & valid
    eval_px = eval_mask & valid

    record = {
        "shape": list(shape),
        "n_valid": int(np.count_nonzero(valid)),
        "n_calib_pixels": int(np.count_nonzero(calib_px)),
        "n_eval_pixels": int(np.count_nonzero(eval_px)),
        "reference": {
            "nearest": _arm_stats(ref_nearest, dem, eval_px),
            "bicubic": _arm_stats(ref_cubic, dem, eval_px),
        },
        "backends": {},
        "skipped_backends": skipped,
        "status": "ok" if np.any(eval_px) else "no_eval_pixels",
    }

    # Cap calibration pixels (deterministic by seed): the 3857 tiles have
    # ~2.5M pixels each; fitting every one costs minutes for no accuracy gain.
    calib_fit_px = _cap_mask(calib_px, calib_cap, rng)
    bicubic_rmse = record["reference"]["bicubic"]["rmse"]

    for name, pred in preds.items():
        backend: dict = {"status": "ok"}
        raw = _arm_stats(pred, dem, eval_px)
        backend.update({
            "raw_r": raw["correlation"], "raw_r_reason": raw["correlation_reason"],
            "raw_mae": raw["mae"], "raw_rmse": raw["rmse"], "raw_bias": raw["bias"],
        })
        fit, fit_error = _fit_on_mask(pred, dem, calib_fit_px)
        if fit is None:
            backend.update({
                "status": "calibration_error", "calibrated_reason": fit_error,
                "cal_r": None, "cal_rmse": None, "cal_mae": None, "cal_bias": None,
                "delta_vs_bicubic": None, "improvement_vs_bicubic": None,
                "scale": None, "offset": None, "inliers": None, "warning": None,
                "estimate_status": None, "polarity_inverted": None,
            })
        elif fit.degenerate:
            backend.update({
                "status": "failed", "calibrated_reason": calib.degenerate_reason(fit),
                "cal_r": None, "cal_rmse": None, "cal_mae": None, "cal_bias": None,
                "delta_vs_bicubic": None, "improvement_vs_bicubic": None,
                "scale": fit.scale, "offset": fit.offset, "inliers": fit.n_inliers,
                "warning": fit.calibration_warning,
                "estimate_status": fit.estimate_status,
                "polarity_inverted": fit.polarity_inverted,
            })
        else:
            cal = _arm_stats(fit.scale * pred.astype("float64") + fit.offset, dem, eval_px)
            delta = None
            improvement = None
            if cal["rmse"] is not None and bicubic_rmse is not None:
                delta = cal["rmse"] - bicubic_rmse
                if bicubic_rmse > _BICUBIC_SELF_REFERENTIAL_M:
                    improvement = (bicubic_rmse - cal["rmse"]) / bicubic_rmse
            backend.update({
                "cal_r": cal["correlation"], "cal_rmse": cal["rmse"],
                "cal_mae": cal["mae"], "cal_bias": cal["bias"],
                "delta_vs_bicubic": delta, "improvement_vs_bicubic": improvement,
                "scale": fit.scale, "offset": fit.offset, "inliers": fit.n_inliers,
                "warning": fit.calibration_warning,
                "estimate_status": fit.estimate_status,
                "polarity_inverted": fit.polarity_inverted,
                "calibrated_reason": None,
            })
        # Within-scene random-split diagnostic: same fit machinery, random
        # pixels instead of spatial blocks - reported, never headlined.
        random_fit_px = _cap_mask(valid, calib_cap, rng)
        rnd_fit, rnd_error = _fit_on_mask(pred, dem, random_fit_px)
        if rnd_fit is None or rnd_fit.degenerate:
            backend.update({
                "random_rmse": None, "random_r": None,
                "random_reason": rnd_error or "degenerate calibration fit",
            })
        else:
            rm = compute_metrics(rnd_fit.scale * rnd_fit.held_relative + rnd_fit.offset,
                                 rnd_fit.held_reference)
            backend.update({
                "random_rmse": rm["rmse"], "random_r": rm["correlation"],
                "random_reason": rm["correlation_reason"],
            })
        record["backends"][name] = backend
    return record


def _site_weighted(tiles, backend: str, key: str):
    """Mean over site means (equal site weight, not equal pixel weight)."""
    per_site: dict = {}
    for t in tiles:
        value = t.get("backends", {}).get(backend, {}).get(key)
        if value is not None:
            per_site.setdefault(t["site_id"], []).append(value)
    site_means = [float(np.mean(v)) for v in per_site.values()]
    return float(np.mean(site_means)) if site_means else None


def _failure_stats(tiles, backend: str) -> dict:
    """Tiles by arm usability; failed/low_confidence/no-metrics all counted.

    failure_rate counts tiles that failed OUTRIGHT (degenerate fit or missing
    metrics); low_confidence tiles carry a warning but still have measurements
    and are tracked separately.
    """
    counts = {"valid": 0, "low_confidence": 0, "failed": 0, "no_metrics": 0}
    for t in tiles:
        if t.get("status") != "ok":
            counts["no_metrics"] += 1
            continue
        info = t.get("backends", {}).get(backend)
        if info is None or info.get("cal_rmse") is None:
            counts["failed" if info and info.get("status") == "failed" else "no_metrics"] += 1
            continue
        status = info.get("estimate_status")
        if status == "valid":
            counts["valid"] += 1
        elif status == "low_confidence":
            counts["low_confidence"] += 1
        else:
            counts["failed"] += 1
    n = len(tiles) or 1
    counts["failure_rate"] = (counts["failed"] + counts["no_metrics"]) / n
    counts["low_confidence_rate"] = counts["low_confidence"] / n
    return counts


def _bootstrap_ci(values, *, iterations: int, seed: int) -> dict:
    """Percentile CI over SITES (values are already site-level)."""
    vals = np.asarray([v for v in values if v is not None], dtype="float64")
    if vals.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None,
                "n_sites": 0, "iters": 0}
    if vals.size < 2 or iterations <= 0:
        return {"mean": float(vals.mean()), "ci_low": None, "ci_high": None,
                "n_sites": int(vals.size), "iters": 0}
    rng = np.random.default_rng(seed)
    n = vals.size
    means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(iterations)])
    return {"mean": float(vals.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "n_sites": int(n), "iters": int(iterations)}


def aggregate_records(tile_records, *, bootstrap_iters: int = 1000,
                      seed: int = 42) -> dict:
    """Site-weighted aggregates + failure counts + site-resampled CIs."""
    backends = sorted({b for t in tile_records for b in t.get("backends", {})})
    sites = sorted({t["site_id"] for t in tile_records})
    strata = sorted({t["stratum"] for t in tile_records})

    def site_values(tiles, backend, key):
        per_site: dict = {}
        for t in tiles:
            value = t.get("backends", {}).get(backend, {}).get(key)
            if value is not None:
                per_site.setdefault(t["site_id"], []).append(value)
        return {site: float(np.mean(v)) for site, v in per_site.items()}

    def arm_summary(tiles, backend):
        warning_counts: dict = {}
        for t in tiles:
            info = t.get("backends", {}).get(backend) or {}
            for token in (info.get("warning") or "").split(";"):
                token = token.strip()
                if token:
                    warning_counts[token] = warning_counts.get(token, 0) + 1
        summary = {
            "cal_rmse": _site_weighted(tiles, backend, "cal_rmse"),
            "cal_mae": _site_weighted(tiles, backend, "cal_mae"),
            "cal_bias": _site_weighted(tiles, backend, "cal_bias"),
            "raw_r": _site_weighted(tiles, backend, "raw_r"),
            "delta_vs_bicubic": _site_weighted(tiles, backend, "delta_vs_bicubic"),
            "improvement_vs_bicubic": _site_weighted(tiles, backend,
                                                     "improvement_vs_bicubic"),
            "n_tiles": len(tiles),
            "n_sites": len({t["site_id"] for t in tiles}),
            "failures": _failure_stats(tiles, backend),
            "warning_counts": warning_counts,
        }
        return summary

    overall_arms = {b: arm_summary(tile_records, b) for b in backends}
    per_stratum = {}
    for stratum in strata:
        tiles = [t for t in tile_records if t["stratum"] == stratum]
        per_stratum[stratum] = {
            "n_tiles": len(tiles),
            "n_sites": len({t["site_id"] for t in tiles}),
            "arms": {b: arm_summary(tiles, b) for b in backends},
        }
    per_site = {}
    for site in sites:
        tiles = [t for t in tile_records if t["site_id"] == site]
        entry = {
            "stratum": tiles[0]["stratum"],
            "split": tiles[0]["split"],
            "n_tiles": len(tiles),
            "arms": {b: arm_summary(tiles, b) for b in backends},
        }
        per_site[site] = entry

    available = [b for b, s in overall_arms.items() if s["cal_rmse"] is not None]
    primary = min(available, key=lambda b: overall_arms[b]["cal_rmse"]) if available else None

    def reference_metric(name):
        per_site: dict = {}
        for t in tile_records:
            value = t.get("reference", {}).get(name, {}).get("rmse")
            if value is not None:
                per_site.setdefault(t["site_id"], []).append(value)
        site_means = [float(np.mean(v)) for v in per_site.values()]
        return float(np.mean(site_means)) if site_means else None

    bootstrap = {}
    for b in backends:
        if b not in available:
            continue
        bootstrap[b] = {
            "cal_rmse": _bootstrap_ci(site_values(tile_records, b, "cal_rmse").values(),
                                      iterations=bootstrap_iters, seed=seed),
            "delta_vs_bicubic": _bootstrap_ci(
                site_values(tile_records, b, "delta_vs_bicubic").values(),
                iterations=bootstrap_iters, seed=seed),
            "raw_r": _bootstrap_ci(site_values(tile_records, b, "raw_r").values(),
                                   iterations=bootstrap_iters, seed=seed),
        }
    return {
        "n_tiles": len(tile_records),
        "n_sites": len(sites),
        "arms": overall_arms,
        "reference": {
            "nearest_rmse": reference_metric("nearest"),
            "bicubic_rmse": reference_metric("bicubic"),
            "note": "reference arms score the stored coarse DEM against itself "
                    "on the shared mask; nonzero only where grids differ",
        },
        "per_stratum": per_stratum,
        "per_site": per_site,
        "primary_arm": primary,
        "bootstrap": bootstrap,
    }


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance(manifest_path, acceptance_path, run_dirs) -> dict:
    """Everything needed to reproduce/attribute a benchmark run."""
    import subprocess

    import rasterio

    commit, dirty = "unknown", None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True,
            text=True, timeout=10).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=_REPO_ROOT, capture_output=True,
            text=True, timeout=10).stdout.strip())
    except Exception:
        pass
    prov = {
        "git_commit": commit,
        "git_dirty": dirty,
        "repo_root": str(_REPO_ROOT),
        "manifest_sha256": _sha256(manifest_path),
        "acceptance_sha256": _sha256(acceptance_path),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "prediction_runs": {},
    }
    for name, run_dir in run_dirs.items():
        info = {"dir": str(run_dir), "exists": Path(run_dir).exists()}
        for fname in ("run.json", "model_info.json", "environment.json"):
            f = Path(run_dir) / fname
            if f.exists():
                try:
                    info[fname] = json.loads(f.read_text(encoding="utf-8"))
                except Exception as exc:
                    info[fname] = f"unreadable: {exc}"
        prov["prediction_runs"][name] = info
    return prov


def _criterion(value, *, threshold, ok, status_reason=None) -> dict:
    """One acceptance check: pass/fail/not_measurable + the value behind it."""
    if value is None:
        return {"status": "not_measurable", "value": None, "threshold": threshold,
                "reason": status_reason or "value not available"}
    return {"status": "pass" if ok(value) else "fail", "value": value,
            "threshold": threshold, "reason": None}


def _verdict_of(checks) -> str:
    statuses: list = []

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("status"), str):
                statuses.append(node["status"])
                return
            for value in node.values():
                walk(value)

    walk(checks)
    if "fail" in statuses:
        return "fail"
    if "not_measurable" in statuses:
        return "not_measurable"
    return "pass"


def evaluate_acceptance(cfg: dict, aggregate: dict, n_test_sites: int) -> dict:
    """Pass/fail per frozen threshold. Thresholds come ONLY from the cfg dict."""
    checks: dict = {}
    arm = aggregate.get("primary_arm")
    if arm is None:
        checks["primary_arm"] = {
            "status": "fail", "value": None, "threshold": None,
            "reason": "no backend produced a usable calibrated arm"}
    else:
        arms = aggregate["arms"]
        min_imp = float(cfg["minimum_improvement"])
        imp = arms[arm]["improvement_vs_bicubic"]
        checks["minimum_improvement"] = _criterion(
            imp, threshold=min_imp, ok=lambda v: v >= min_imp,
            status_reason="bicubic baseline is self-referential (RMSE ~ 0): "
                          "improvement needs an independent/higher-resolution "
                          "reference (manifest column reference_hr_path is empty)")
        per_stratum = {}
        for stratum, need in (cfg.get("per_stratum_improvement") or {}).items():
            need_f = float(need)
            s_imp = (aggregate["per_stratum"].get(stratum, {}).get("arms", {})
                     .get(arm, {}).get("improvement_vs_bicubic"))
            per_stratum[stratum] = _criterion(
                s_imp, threshold=need_f, ok=lambda v, n=need_f: v >= n,
                status_reason="same self-referential bicubic baseline as the overall "
                              "improvement check")
        checks["per_stratum_improvement"] = per_stratum
        raw_r = arms[arm]["raw_r"]
        min_r = float(cfg["min_raw_correlation"])
        checks["min_raw_correlation"] = _criterion(
            raw_r, threshold=min_r, ok=lambda v: v >= min_r,
            status_reason="no calibrated tile produced a defined raw correlation")
        max_fail = float(cfg["max_failure_rate"])
        checks["max_failure_rate"] = _criterion(
            arms[arm]["failures"]["failure_rate"], threshold=max_fail,
            ok=lambda v: v <= max_fail)
    min_sites = float(cfg["min_test_sites"])
    checks["min_test_sites"] = _criterion(
        float(n_test_sites), threshold=min_sites, ok=lambda v: v >= min_sites)
    return {"checks": checks, "verdict": _verdict_of(checks), "primary_arm": arm}


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


_TILE_CSV_COLUMNS = (
    "tile_id", "site_id", "stratum", "split", "backend", "status", "n_eval_pixels",
    "raw_r", "raw_r_reason", "raw_mae", "raw_rmse", "raw_bias",
    "cal_r", "cal_rmse", "cal_mae", "cal_bias", "scale", "offset", "inliers",
    "warning", "estimate_status", "calibrated_reason", "polarity_inverted",
    "delta_vs_bicubic", "improvement_vs_bicubic", "random_rmse", "random_r",
    "ref_bicubic_rmse", "ref_nearest_rmse", "notes",
)


def _write_per_tile_csv(path: Path, tile_records) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TILE_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for t in tile_records:
            base = {
                "tile_id": t.get("tile_id"), "site_id": t.get("site_id"),
                "stratum": t.get("stratum"), "split": t.get("split"),
                "n_eval_pixels": t.get("n_eval_pixels"),
                "ref_bicubic_rmse": t.get("reference", {}).get("bicubic", {}).get("rmse"),
                "ref_nearest_rmse": t.get("reference", {}).get("nearest", {}).get("rmse"),
                "notes": t.get("notes", ""),
            }
            rows = [{**base, **info, "backend": name}
                    for name, info in (t.get("backends") or {}).items()]
            rows += [{**base, "backend": name, "status": "grid_mismatch",
                      "raw_r_reason": reason}
                     for name, reason in (t.get("skipped_backends") or {}).items()]
            if not rows:
                rows.append({**base, "backend": "",
                             "status": t.get("status", "no_predictions")})
            for row in rows:
                writer.writerow({c: _fmt(row.get(c)) for c in _TILE_CSV_COLUMNS})


def _md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _write_markdown(path: Path, payload) -> None:
    meta, agg, acc = payload["meta"], payload["aggregate"], payload["acceptance"]
    arm = agg["primary_arm"]
    lines = [
        "# DepthWizard benchmark",
        "",
        f"- git `{meta['git_commit']}` (dirty: {meta['git_dirty']}), rasterio "
        f"{meta['rasterio']} / GDAL {meta['gdal']}",
        f"- manifest `{payload['manifest']['path']}` sha256 "
        f"`{meta['manifest_sha256'][:12]}`; acceptance `{acc['path']}` sha256 "
        f"`{meta['acceptance_sha256'][:12]}`",
        f"- split `{payload['run']['split']}`: {agg['n_tiles']} tiles / "
        f"{agg['n_sites']} sites; block={payload['run']['block']} "
        f"guard={payload['run']['guard']} calib_cap={payload['run']['calib_cap']} "
        f"seed={payload['run']['seed']}",
        f"- primary arm (lowest site-weighted RMSE): `{arm}`",
        "",
        "## Overall (site-weighted means)",
        "",
        _md_table(
            ["arm", "cal RMSE (m)", "cal MAE (m)", "bias (m)", "raw r",
             "delta vs bicubic (m)", "improvement", "valid/low/failed/no_metrics",
             "failure rate"],
            [[name, s["cal_rmse"], s["cal_mae"], s["cal_bias"], s["raw_r"],
              s["delta_vs_bicubic"], s["improvement_vs_bicubic"],
              "{valid}/{low_confidence}/{failed}/{no_metrics}".format(
                  **s["failures"]),
              s["failures"]["failure_rate"]]
             for name, s in agg["arms"].items()]),
        "",
        f"Reference arms: nearest RMSE {_fmt(agg['reference']['nearest_rmse'])} m, "
        f"bicubic RMSE {_fmt(agg['reference']['bicubic_rmse'])} m (stored coarse DEM "
        "scored against itself on the shared mask).",
        "",
    ]
    if arm is not None:
        lines += [
            "## Per stratum (primary arm)",
            "",
            _md_table(
                ["stratum", "tiles", "sites", "cal RMSE (m)", "raw r",
                 "improvement", "failure rate"],
                [[st, v["n_tiles"], v["n_sites"],
                  v["arms"][arm]["cal_rmse"], v["arms"][arm]["raw_r"],
                  v["arms"][arm]["improvement_vs_bicubic"],
                  v["arms"][arm]["failures"]["failure_rate"]]
                 for st, v in agg["per_stratum"].items()]),
            "",
            "## Per site (primary arm)",
            "",
            _md_table(
                ["site", "stratum", "split", "tiles", "cal RMSE (m)", "raw r",
                 "delta vs bicubic (m)", "failure rate"],
                [[site, v["stratum"], v["split"], v["n_tiles"],
                  v["arms"][arm]["cal_rmse"], v["arms"][arm]["raw_r"],
                  v["arms"][arm]["delta_vs_bicubic"],
                  v["arms"][arm]["failures"]["failure_rate"]]
                 for site, v in agg["per_site"].items()]),
            "",
        ]
    lines += ["## Bootstrap (95% CI over sites)", ""]
    boot_rows = []
    for name, block in agg["bootstrap"].items():
        for metric, ci in block.items():
            boot_rows.append([name, metric, ci["mean"], ci["ci_low"], ci["ci_high"],
                              ci["n_sites"], ci["iters"]])
    lines += [_md_table(["arm", "metric", "mean", "ci low", "ci high", "n sites",
                         "iters"], boot_rows), ""]
    if arm is not None and agg["arms"][arm]["warning_counts"]:
        lines += ["## Warning counts (primary arm)", "",
                  _md_table(["warning", "tiles"],
                            sorted(agg["arms"][arm]["warning_counts"].items())), ""]
    lines += [f"## Acceptance verdict: **{acc['verdict'].upper()}**", ""]
    check_rows = []

    def walk(prefix, node):
        if isinstance(node, dict) and isinstance(node.get("status"), str):
            check_rows.append([prefix, node["status"], node.get("value"),
                               node.get("threshold"), node.get("reason") or ""])
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)

    walk("", acc["checks"])
    lines += [_md_table(["check", "status", "value", "threshold", "reason"],
                        check_rows), ""]
    caveats = []
    total_sites = len({t["site_id"] for t in payload["tiles"]})
    thin = sorted(name for name, block in agg["bootstrap"].items()
                  if "cal_rmse" in block and block["cal_rmse"]["n_sites"] < total_sites)
    if thin:
        caveats.append(
            "arms cover fewer sites than the split has where a calibration fit "
            "failed outright (bootstrap 'n sites' column: "
            + ", ".join(f"{name} {agg['bootstrap'][name]['cal_rmse']['n_sites']}"
                        f"/{total_sites}" for name in thin)
            + "); failed tiles stay in the failure counts.")
    bic = agg["reference"]["bicubic_rmse"]
    if bic is not None and bic < _BICUBIC_SELF_REFERENTIAL_M:
        caveats.append(
            "the bicubic arm is self-referential on this manifest (RMSE ~ 0): "
            "'improvement vs bicubic' is NOT measurable until an independent or "
            "higher-resolution reference is populated (reference_hr_path column).")
    leak = [t["tile_id"] for t in payload["tiles"]
            if "FROM A TEST SITE" in (t.get("notes") or "")]
    if leak and "finetuned" in agg["arms"]:
        caveats.append(
            "finetuned backend: its decoder fine-tuned across most tiles; manifest "
            "notes flag test-site validation tiles " + ", ".join(leak) + " - treat "
            "finetuned test-split numbers as contaminated diagnostics, not "
            "generalization evidence.")
    tilezen = [t["tile_id"] for t in payload["tiles"]
               if "Tilezen" in (t.get("reference_product") or "")]
    if tilezen:
        caveats.append(
            f"{len(tilezen)} tiles use the Tilezen terrarium fallback reference "
            "product with an unpublished vertical datum (manifest notes): their "
            "absolute errors are not comparable with Copernicus GLO-30 tiles.")
    caveats.append(
        "raw-arm MAE/RMSE are in relative-depth units (only correlations and the "
        "calibrated arms are comparable across arms); delta vs bicubic is defined "
        "for the calibrated arm only.")
    caveats.append(
        "the random-split columns are the within-scene diagnostic (optimistic by "
        "construction); headline numbers are spatial-block with guard band.")
    caveats.append(
        "failed / low_confidence tiles remain in every count; the failure rate "
        "counts tiles that failed outright (degenerate fit or missing metrics), "
        "while warning-carrying tiles are listed in the counts column and the "
        "warning-count table above.")
    failed = sorted({t["tile_id"] for t in payload["tiles"]
                     if any((i or {}).get("status") in ("failed", "calibration_error")
                            for i in (t.get("backends") or {}).values())})
    if failed:
        caveats.append(
            "tiles that hit the calibration fit's degenerate/fallback path in this "
            "run: " + ", ".join(failed) + " (see per_tile.csv calibrated_reason); "
            "they stay in every count.")
    lines += ["## Caveats", ""] + [f"- {c}" for c in caveats] + [""]
    if payload["manifest"]["warnings"]:
        lines += ["## Manifest warnings", ""] + [
            f"- {w}" for w in payload["manifest"]["warnings"]] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(args) -> dict:
    """The `benchmark` CLI body: manifest -> predictions -> artifacts."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required for the benchmark CLI "
                         "(already present transitively; make it a direct dep "
                         "if this fires)") from exc

    from . import dem_source

    manifest_path = Path(args.manifest)
    rows, validation = load_manifest(manifest_path, guard_km=args.guard_km)
    for warning in validation["warnings"]:
        print(f"[manifest warning] {warning}")

    acceptance_path = Path(args.acceptance)
    if not acceptance_path.exists():
        raise SystemExit(f"acceptance file not found: {acceptance_path}")
    cfg = yaml.safe_load(acceptance_path.read_text(encoding="utf-8"))
    required = ("minimum_improvement", "per_stratum_improvement", "max_failure_rate",
                "min_raw_correlation", "min_test_sites", "bootstrap_iters", "seed")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"acceptance file {acceptance_path} is missing: {missing}")

    selected = [r for r in rows if args.split == "all" or r["split"] == args.split]
    if not selected:
        raise SystemExit(f"no tiles in split {args.split!r}")

    run_dirs: dict = {}
    for name, rel in _DEFAULT_PREDICTION_RUNS.items():
        path = Path(rel)
        run_dirs[name] = path if path.is_absolute() else _REPO_ROOT / path
    for override in args.predictions or []:
        name, sep, path = override.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--predictions expects BACKEND=DIR, got {override!r}")
        run_dirs[name] = Path(path)
    if args.fusion_pred_dir:
        run_dirs["fusion"] = Path(args.fusion_pred_dir)

    availability = {name: {"dir": str(d), "tiles_present": 0, "tiles_missing": 0}
                    for name, d in run_dirs.items()}
    if "fusion" not in run_dirs:
        availability["fusion"] = {
            "status": "unavailable", "provided": False,
            "reason": "phase 3 fusion predictions not provided (--fusion-pred-dir)"}

    tiles_root = Path(args.tiles_root)
    tile_records = []
    for row in selected:
        tile_id = row["tile_id"]
        meta_cols = {k: row.get(k) for k in (
            "tile_id", "site_id", "region", "split", "stratum", "sensor", "crs",
            "gsd", "reference_product", "reference_res_m", "reference_hr_path")}
        dem_path = tiles_root / row["stratum"] / f"{tile_id}_dem.tif"
        if not dem_path.exists():
            tile_records.append({
                **meta_cols, "notes": row.get("notes", ""),
                "status": "missing_reference", "shape": None, "n_valid": 0,
                "n_calib_pixels": 0, "n_eval_pixels": 0,
                "reference": {"nearest": {}, "bicubic": {}},
                "backends": {}, "skipped_backends": {}})
            print(f"[skip] {tile_id}: reference DEM not found at {dem_path}")
            continue
        dem, meta = dem_source._read_local_geotiff(str(dem_path))
        georef = (meta.get("transform"), meta.get("crs"),
                  meta.get("transform"), meta.get("crs"))
        preds = {}
        for name, run_dir in run_dirs.items():
            fname = f"{tile_id}_relative.npy"
            if name == "fusion" and not (run_dir / "predictions" / fname).exists():
                fname = f"{tile_id}_residual.npy"
            npy = run_dir / "predictions" / fname
            if npy.exists():
                preds[name] = np.load(npy)
                availability[name]["tiles_present"] += 1
            else:
                availability[name]["tiles_missing"] += 1
        record = evaluate_tile_arrays(dem, preds, georef=georef, block=args.block,
                                      guard=args.guard, calib_cap=args.calib_cap,
                                      seed=args.seed)
        record.update(meta_cols)
        record["notes"] = row.get("notes", "")
        tile_records.append(record)
    for name, info in availability.items():
        if "tiles_present" in info:
            info["provided"] = True
            info["status"] = ("unavailable" if info["tiles_present"] == 0
                              else "partial" if info["tiles_missing"] else "complete")

    aggregate = aggregate_records(tile_records,
                                  bootstrap_iters=int(cfg["bootstrap_iters"]),
                                  seed=int(cfg["seed"]))
    n_test_sites = len({r["site_id"] for r in rows if r["split"] == "test"})
    acceptance = evaluate_acceptance(cfg, aggregate, n_test_sites)

    split_counts: dict = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    payload = {
        "meta": _provenance(manifest_path, acceptance_path, run_dirs),
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path),
                     "n_tiles_total": len(rows), "splits": split_counts,
                     "guard_km": args.guard_km, "warnings": validation["warnings"]},
        "run": {"split": args.split, "tiles_evaluated": len(tile_records),
                "block": args.block, "guard": args.guard,
                "calib_cap": args.calib_cap, "seed": args.seed},
        "acceptance": {"path": str(acceptance_path),
                       "sha256": _sha256(acceptance_path), "config": cfg,
                       "checks": acceptance["checks"],
                       "verdict": acceptance["verdict"],
                       "primary_arm": acceptance["primary_arm"]},
        "availability": availability,
        "aggregate": aggregate,
        "tiles": tile_records,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_per_tile_csv(out_dir / "per_tile.csv", tile_records)
    _write_markdown(out_dir / "benchmark.md", payload)
    print(f"[benchmark] verdict={acceptance['verdict']} "
          f"primary_arm={aggregate['primary_arm']} sites={aggregate['n_sites']} "
          f"tiles={aggregate['n_tiles']} -> {out_dir}")
    return payload


def _build_benchmark_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline.evaluation",
        description="DepthWizard accuracy evaluation + model benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark",
                           help="run the benchmark on cached predictions")
    default = _REPO_ROOT
    bench.add_argument("--manifest", default=str(default / "server" / "data" / "manifest.csv"))
    bench.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    bench.add_argument("--out", default=str(default / "results"))
    bench.add_argument("--tiles-root", default=str(default / "training_data"))
    bench.add_argument("--acceptance",
                       default=str(default / "server" / "config" / "acceptance.yaml"))
    bench.add_argument("--predictions", action="append", default=[],
                       metavar="BACKEND=DIR",
                       help="override a backend's prediction run dir (repeatable)")
    bench.add_argument("--fusion-pred-dir", default=None,
                       help="run dir with fusion predictions (phase 3; optional)")
    bench.add_argument("--block", type=int, default=180)
    bench.add_argument("--guard", type=int, default=8)
    bench.add_argument("--calib-cap", type=int, default=2_000_000)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--guard-km", type=float, default=10.0)
    return parser


def main(argv=None) -> int:
    args = _build_benchmark_parser().parse_args(argv)
    if args.command == "benchmark":
        run_benchmark(args)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())