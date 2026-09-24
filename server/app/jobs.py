"""Background job orchestration.

A single-worker thread processes jobs queued in SQLite. The HTTP layer creates
a job row then enqueues it; the worker runs the full pipeline
(depth inference -> calibration -> export) and writes the result row.

/status polling reads progress from an in-memory dict (per-process), while the
authoritative job status (queued/running/done/failed) lives in SQLite.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import config, db
from .pipeline import (uploader, depth, calibration as calib,
                       dem_source, exporter, validate as vmod)
from .pipeline import artifacts

_STAGE_PROGRESS = {
    "loading": 0.05,
    "running_depth_model": 0.5,
    "calibrating": 0.7,
    "building_terrain": 0.9,
    "done": 1.0,
}

_queue: "list[str]" = []
_queue_cond = threading.Condition()
_worker_started = False
_progress_cache: dict = {}


def start_worker() -> None:
    global _worker_started
    with _queue_cond:
        if _worker_started:
            return
        _worker_started = True
        threading.Thread(target=_worker_loop, daemon=True, name="dw-worker").start()


def enqueue(job_id: str) -> None:
    with _queue_cond:
        _queue.append(job_id)
        _queue_cond.notify()


def get_progress(job_id: str) -> tuple:
    return _progress_cache.get(job_id, (None, None))


def set_progress(job_id: str, progress: float, stage: str) -> None:
    _progress_cache[job_id] = (progress, stage)


def _worker_loop() -> None:
    while True:
        with _queue_cond:
            while not _queue:
                _queue_cond.wait()
            job_id = _queue.pop(0)
        try:
            _run_job(job_id)
        except Exception:  # pragma: no cover - defensive
            try:
                db.set_job_status(job_id, "failed", error="Unhandled worker error")
            except Exception:
                pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if job is None:
        return
    upload = db.get_upload(job["upload_id"])
    if upload is None:
        db.set_job_status(job_id, "failed", error="Upload not found")
        return

    branch = job["branch"]
    try:
        set_progress(job_id, _STAGE_PROGRESS["loading"], "loading")
        file_path = Path(upload["file_path"])
        arr, meta = uploader.load_raster(file_path)

        # ---------- relative depth -------------------------------------- #
        set_progress(job_id, _STAGE_PROGRESS["running_depth_model"], "running_depth_model")
        rgb = _to_rgb(arr)
        rdsm = depth.infer_relative_dsm(rgb)
        # depth/relative maps are always 2D; squeeze any spurious channel dims
        if rdsm.ndim == 3:
            if rdsm.shape[0] == 1:
                rdsm = rdsm[0]
            elif rdsm.shape[2] == 1:
                rdsm = rdsm[:, :, 0]
            else:
                rdsm = np.mean(rdsm, axis=2)

        min_elev = max_elev = None
        cell_size = None
        dsm_geotiff_path = None
        is_geo = bool(upload["is_georeferenced"])

        if branch == "absolute_dsm" and is_geo:
            set_progress(job_id, _STAGE_PROGRESS["calibrating"], "calibrating")
            result_dir = config.FILES_DIR / job_id
            elevation, cell_size, dsm_geotiff_path = _calibrate_absolute(
                rdsm, upload, meta, result_dir
            )
            min_elev = float(elevation.min())
            max_elev = float(elevation.max())
        else:
            # relative branch: scale normalized rDSM (0..1) into legible meters
            elevation = rdsm * float(config.RELATIVE_ELEVATION_RANGE_M)
            min_elev = float(elevation.min())
            max_elev = float(elevation.max())

        # ---------- export ---------------------------------------------- #
        set_progress(job_id, _STAGE_PROGRESS["building_terrain"], "building_terrain")
        result_dir = config.FILES_DIR / job_id
        result_dir.mkdir(parents=True, exist_ok=True)

        heightmap_path = str(result_dir / "heightmap.png")
        texture_path = str(result_dir / "texture.png")
        exporter.export_heightmap(elevation, Path(heightmap_path))
        exporter.export_texture(rgb, Path(texture_path))

        world_width, world_depth = exporter.compute_world_dimensions(
            elevation.shape[0], elevation.shape[1], cell_size,
            bounds=upload["bounds"], crs=upload["crs"]
        )

        db.create_result(
            job_id=job_id,
            heightmap_path=heightmap_path,
            texture_path=texture_path,
            dsm_geotiff_path=dsm_geotiff_path,
            min_elev=min_elev,
            max_elev=max_elev,
            cell_size=cell_size,
            world_width=world_width,
            world_depth=world_depth,
        )
        db.set_job_status(job_id, "done")
        set_progress(job_id, 1.0, "done")
    except Exception as e:
        db.set_job_status(job_id, "failed", error=str(e))
        set_progress(job_id, 0.0, "failed")


def _calibrate_absolute(rdsm: np.ndarray, upload: dict, meta: dict, result_dir: Path):
    """Calibrate relative->absolute and export a GeoTIFF.

    Returns (abs_dsm_2d, cell_size_m, geotiff_relative_path).
    """
    from rasterio.transform import from_origin

    bounds = upload["bounds"] or meta.get("bounds")
    crs = upload["crs"] or meta.get("crs")
    if not bounds or not crs:
        raise ValueError("Georeferenced upload is missing bounds/CRS")

    ref, _ = dem_source.fetch_reference_dem(bounds, crs)
    if ref.shape[:2] != rdsm.shape:
        ref = calib.resample_to_grid(ref, rdsm.shape)

    fit = calib.fit_affine(rdsm, ref)
    abs_dsm = calib.apply_fit(rdsm, fit)
    # Persist the fit + held-out validation set so /validate can score the same
    # 20% of pixels (never the calibration's training 80%) - leak-proof parity
    # with POST /evaluate.
    calib.save_fit(fit, result_dir / "held_out.npz")

    minx, miny, maxx, maxy = bounds
    cell_size = abs((maxx - minx) / rdsm.shape[1])
    if cell_size <= 0:
        cell_size = None

    result_dir.mkdir(parents=True, exist_ok=True)
    geotiff = result_dir / "dsm.tif"
    transform = meta.get("transform") if meta.get("transform") is not None \
        else from_origin(minx, maxy, cell_size or 1.0, cell_size or 1.0)
    exporter.export_geotiff(abs_dsm, crs, transform, bounds, geotiff)

    return abs_dsm, cell_size, str(geotiff)


def run_validation(job_id: str, *, save_artifacts: bool = False) -> dict:
    """Compare the produced absolute DSM against the reference DEM and store metrics.

    save_artifacts:
        Opt-in prediction-artifact export of the FINAL persisted prediction
        (the absolute DSM the renderer consumed). Raw/relative stages are NOT
        persisted by the job pipeline, so only the final prediction is written
        (the artifacts layer never re-runs the model).
    """
    result = db.get_result_by_job(job_id)
    if result is None or not result.get("dsm_geotiff_path"):
        raise ValueError("No absolute DSM available for validation")

    job = db.get_job(job_id)
    upload = db.get_upload(job["upload_id"])
    bounds = upload["bounds"]
    crs = upload["crs"]
    if not bounds or not crs:
        raise ValueError("Georeferenced metadata required for validation")

    import rasterio

    with rasterio.open(result["dsm_geotiff_path"]) as src:
        pred = src.read(1).astype("float32")
        pred_transform = src.transform
        pred_crs = str(src.crs) if src.crs else None

    ref, rmeta = dem_source.fetch_reference_dem(bounds, crs)
    ref = calib.resample_to_grid(ref, pred.shape)

    # Leak-proof calibration guarantee (same as POST /evaluate): score ONLY the
    # held-out 20% that fit_affine reserved during processing and persisted to
    # held_out.npz (fixed split seed => the same pixels every run). Jobs from
    # before holdout persistence existed fall back to the whole-grid comparison
    # rather than failing.
    fit = calib.load_fit(Path(result["heightmap_path"]).parent / "held_out.npz")
    degenerate_calibration = bool(fit is not None and fit.degenerate)
    if fit is not None and not fit.degenerate \
            and fit.held_relative is not None and fit.held_relative.size >= 2:
        held_pred = fit.scale * fit.held_relative + fit.offset
        held_truth = fit.held_reference
        metrics = vmod.compute_metrics(held_pred, held_truth)
        held_out_count = int(metrics["n"])
    else:
        metrics = vmod.compute_metrics(pred, ref)
        held_out_count = None

    # Calibration health surfaced from the persisted fit. Legacy jobs with no
    # held_out.npz are None here (no fit), but correlation_reason still applies.
    calib_status = fit.calibration_status if fit is not None else None
    calib_warning = fit.calibration_warning if fit is not None else None
    calib_scale = fit.scale if fit is not None else None

    if save_artifacts:
        artifact_metrics = {k: v for k, v in {
            "mae": metrics["mae"], "rmse": metrics["rmse"],
            "correlation": metrics["correlation"],
            "correlation_reason": metrics.get("correlation_reason"),
            "calibration_status": calib_status,
            "calibration_warning": calib_warning,
            "calibration_scale": calib_scale,
            "calibration_offset": (fit.offset if fit is not None else None),
        }.items() if v is not None}
        artifacts.export_prediction_artifacts(
            out_root=config.OUTPUT_DIR,
            backend=depth.backend_slug(),
            model_identifier=depth.model_identifier(),
            input_filename=upload["original_filename"] or f"{job_id}.tif",
            job_id=job_id,
            calibrated=pred,
            ground_truth=ref,
            metrics=artifact_metrics,
            crs=pred_crs,
            transform=pred_transform,
            bounds=bounds,
        )

    heat_path = Path(result["heightmap_path"]).parent / "diff_heatmap.png"
    vmod.render_diff_heatmap(pred, ref, heat_path)

    # Record the source actually used (OpenTopography/Copernicus/local), not a
    # hardcoded label. No landscape classifier exists in this codebase, so
    # landscape_type stays null rather than a fabricated value.
    reference_source = rmeta.get("source") or "SRTM_30m"
    db.create_evaluation(
        result_id=result["id"],
        reference_source=reference_source,
        landscape_type=None,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        correlation=metrics["correlation"],
    )

    return {
        "reference_source": reference_source,
        "landscape_type": None,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "correlation": metrics["correlation"],
        "correlation_reason": metrics.get("correlation_reason"),
        "diff_heatmap_url": f"/files/{job_id}/diff_heatmap.png",
        "held_out_pixel_count": held_out_count,
        "degenerate_calibration": degenerate_calibration,
        "calibration_reason": calib.degenerate_reason(fit) if degenerate_calibration else None,
        "calibration_status": calib_status,
        "calibration_warning": calib_warning,
        "calibration_scale": calib_scale,
        "calibration_offset": (fit.offset if fit is not None else None),
    }


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure H,W,3 uint8 RGB for the depth model."""
    if arr.ndim == 3 and arr.shape[0] == 3:
        out = arr.transpose(1, 2, 0)
    elif arr.ndim == 3 and arr.shape[0] > 3:
        out = arr[:3].transpose(1, 2, 0)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        out = np.repeat(arr[0, :, :][:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        out = arr
    elif arr.ndim == 2:
        out = np.stack([arr] * 3, axis=-1)
    else:
        out = np.repeat(arr[0, :, :][:, :, None], 3, axis=2)
    if out.max() <= 1.05:
        out = out * 255.0
    return np.clip(out, 0, 255).astype("uint8")
