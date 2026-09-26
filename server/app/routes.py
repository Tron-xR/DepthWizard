"""HTTP route handlers for the DepthWizard FastAPI server."""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from . import config, db, jobs
from .errors import DepthWizardError
from .pipeline import uploader, depth, evaluation, artifacts, exporter
from .schemas import (
    ErrorResponse,
    EvaluationResponse,
    HealthResponse,
    ProcessResponse,
    ResultResponse,
    StatusResponse,
    UploadResponse,
    ValidationResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    try:
        content = await file.read()
        info = uploader.ingest_upload(content, file.filename or "upload")
    except DepthWizardError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse(error=e.error_code, message=e.message).model_dump())

    upload_id = db.create_upload(
        original_filename=file.filename or "upload",
        file_path=info["file_path"],
        file_type=info["file_type"],
        is_georeferenced=info["is_georeferenced"],
        crs=info.get("crs"),
        bounds=info.get("bounds"),
    )["id"]
    return UploadResponse(
        upload_id=upload_id,
        is_georeferenced=info["is_georeferenced"],
        crs=info.get("crs"),
        bounds=info.get("bounds"),
    )


@router.get("/preview/{upload_id}")
def preview(upload_id: str) -> Response:
    """Low-res PNG thumbnail of an uploaded file (Unity pre-Process preview).

    Renders the already-ingested working raster via uploader.render_preview_png
    (rasterio for GeoTIFF, Pillow for PNG/JPG). Purely cosmetic; never feeds
    the pipeline or metrics.
    """
    up = db.get_upload(upload_id)
    if up is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Upload not found").model_dump())
    try:
        png = uploader.render_preview_png(Path(up["file_path"]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=ErrorResponse(error="preview_failed", message=f"Preview unavailable: {e}").model_dump())
    return Response(content=png, media_type="image/png")


@router.post("/process/{upload_id}", response_model=ProcessResponse)
def process(upload_id: str) -> ProcessResponse:
    upload = db.get_upload(upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Upload not found").model_dump())

    branch = "absolute_dsm" if upload["is_georeferenced"] else "rdsm"
    job = db.create_job(upload["id"], branch)
    jobs.enqueue(job["id"])
    return ProcessResponse(job_id=job["id"], status="queued")


@router.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str) -> StatusResponse:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Job not found").model_dump())
    progress, stage = jobs.get_progress(job_id)
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=progress if job["status"] in ("running", "queued", "done", "failed") else progress,
        stage=stage,
    )


@router.get("/result/{job_id}", response_model=ResultResponse)
def result(job_id: str) -> ResultResponse:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Job not found").model_dump())
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=ErrorResponse(error="not_ready", message="Job is not done").model_dump())

    res = db.get_result_by_job(job_id)
    upload = db.get_upload(job["upload_id"])
    return _result_response(res, upload)


def _result_response(res: dict, upload: dict) -> ResultResponse:
    job_id = res["job_id"]
    return ResultResponse(
        heightmap_url=f"/files/{job_id}/heightmap.png",
        texture_url=f"/files/{job_id}/texture.png",
        dsm_geotiff_url=(f"/files/{job_id}/dsm.tif" if res.get("dsm_geotiff_path") else None),
        min_elev=res.get("min_elev"),
        max_elev=res.get("max_elev"),
        cell_size=res.get("cell_size"),
        world_width=res["world_width"],
        world_depth=res["world_depth"],
        is_georeferenced=bool(upload["is_georeferenced"]),
    )


@router.get("/dem-view/{job_id}")
def dem_view(job_id: str) -> Response:
    """Grayscale DEM overlay PNG of a finished job's calibrated elevation.

    Renders the same true-scale elevation array the Unity mesh consumed
    (full-precision dsm.tif when archived, else the 8-bit heightmap
    de-normalized to meters via the stored min/max), min-max normalized to
    this job's own range. Never exaggerated. Purely cosmetic; like /preview it
    feeds nothing back into the pipeline.
    """
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Job not found").model_dump())
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=ErrorResponse(error="not_ready", message="Job is not done").model_dump())
    res = db.get_result_by_job(job_id)
    if res is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Result not found").model_dump())
    try:
        png = exporter.dem_view_png(res["heightmap_path"], res.get("dsm_geotiff_path"),
                                    res.get("min_elev"), res.get("max_elev"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=ErrorResponse(error="dem_view_failed", message=f"DEM view unavailable: {e}").model_dump())
    return Response(content=png, media_type="image/png")


@router.get("/export-dsm/{job_id}")
def export_dsm(job_id: str) -> FileResponse:
    """Downloadable GeoTIFF of a job's real DSM (absolute/georeferenced branch).

    The same true-scale elevation array the Unity mesh and /dem-view consume
    (no vertical exaggeration), float32 meters with the original upload's
    CRS/transform, plus metadata tagging it as a model-estimated surface.
    Relative (non-georeferenced) jobs have no CRS or real scale, so they get a
    clear 4xx instead of a misleading raster.
    """
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Job not found").model_dump())
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=ErrorResponse(error="not_ready", message="Job is not done").model_dump())
    res = db.get_result_by_job(job_id)
    if res is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Result not found").model_dump())
    if not res.get("dsm_geotiff_path"):
        raise HTTPException(status_code=400, detail=ErrorResponse(
            error="dsm_export_requires_georeferenced",
            message="DSM export needs a georeferenced input (GPS/EXIF tags); this was a relative-only run with no CRS or real elevation.").model_dump())
    try:
        upload = db.get_upload(job["upload_id"])
        out = exporter.export_dsm_geotiff(
            Path(res["dsm_geotiff_path"]),
            Path(config.FILES_DIR) / job_id / "dsm_export.tif",
            job_id=job_id,
            input_filename=upload["original_filename"] or "upload",
            backend=depth.backend_slug(),
            model=depth.model_identifier(),
            min_elev=res.get("min_elev"),
            max_elev=res.get("max_elev"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=ErrorResponse(error="dsm_export_failed", message=f"DSM export failed: {e}").model_dump())
    return FileResponse(out, media_type="image/tiff", filename="dsm_export.tif")


@router.get("/validate/{job_id}", response_model=ValidationResponse)
def validate(job_id: str, save_artifacts: bool = False) -> ValidationResponse:
    """Validate a processed job's absolute DSM against the reference DEM.

    save_artifacts:
        opt-in prediction-artifact export (final persisted DSM only) under
        DEPTHWIZARD_OUTPUT_DIR. Off by default.
    """
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(error="not_found", message="Job not found").model_dump())
    try:
        data = jobs.run_validation(job_id, save_artifacts=save_artifacts)
    except DepthWizardError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse(error=e.error_code, message=e.message).model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=ErrorResponse(error="validation_failed", message=str(e)).model_dump())
    return ValidationResponse(**data)


# -------- ad hoc RGB + DEM accuracy evaluation ---------------------------- #
_SAVE_SUFFIX = {".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpg",
                ".tif": ".tif", ".tiff": ".tif"}


def _save_to_cache(content: bytes, filename: str) -> Path:
    """Persist uploaded bytes to the cache, keeping the native extension.

    Unlike uploader.ingest_upload we do NOT downsample: for a pixel-aligned
    RGB+DEM pair we must preserve the exact uploaded grid so the comparison is
    meaningful.
    """
    suffix = _SAVE_SUFFIX.get(Path(filename).suffix.lower())
    if suffix is None:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(error="unsupported_file_type",
                                 message="Only PNG, JPG, and GeoTIFF are supported.").model_dump(),
        )
    dest = config.CACHE_DIR / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(content)
    return dest


def _to_2d_surface(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(image: UploadFile = File(...),
                   dem: UploadFile = File(...),
                   mode: str = Form("calibrated"),
                   include_visualization: bool = Form(False),
                   save_artifacts: bool = Form(False)) -> EvaluationResponse:
    """Evaluate a matched RGB + ground-truth DEM pair (no georeferencing needed).

    mode:
      "calibrated" (default) - fit scale/offset on 80% of valid pixels, report
                               MAE/RMSE/correlation on the held-out 20% only.
      "relative"             - raw relative-depth correlation only (MAE/RMSE null).

    include_visualization:
      opt-in 3-panel comparison PNG (RGB | ground truth | calibrated prediction,
      shared elevation colormap) saved to /files/<id>/comparison.png, returned
      as comparison_image_url. Only produced in calibrated, non-degenerate mode.

    save_artifacts:
      opt-in prediction-artifact export under DEPTHWIZARD_OUTPUT_DIR
      (<root>/<backend>/<input-stem>/): exact-float predicted_depth.npy,
      raw/relative/calibrated stage .npy's, a visualization-only .png, a
      georeferenced single-band float .tif when the input was georeferenced,
      and metrics.json. Off by default; never affects the reported metrics.
    """
    if mode not in ("calibrated", "relative"):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(error="bad_request",
                                 message="mode must be 'calibrated' or 'relative'.").model_dump(),
        )

    import rasterio

    try:
        image_path = _save_to_cache(await image.read(), image.filename or "image.png")
        dem_path = _save_to_cache(await dem.read(), dem.filename or "dem.tif")

        rgb_arr, rgb_meta = uploader.load_raster(image_path)
        rgb = jobs._to_rgb(rgb_arr)
        capture: dict = {}
        rdsm = depth.infer_relative_dsm(rgb, capture=capture)
        rdsm = _to_2d_surface(np.asarray(rdsm))

        with rasterio.open(dem_path) as src:
            dem_2d = src.read(1).astype("float64")
            nodata = src.nodata
            dem_transform, dem_crs = src.transform, str(src.crs) if src.crs else None

        dem_2d = evaluation.align_dem_to_prediction(
            rdsm, dem_2d,
            predicted_transform=rgb_meta.get("transform"),
            predicted_crs=rgb_meta.get("crs"),
            dem_transform=dem_transform,
            dem_crs=dem_crs,
        )
    except DepthWizardError as e:
        raise HTTPException(status_code=400,
                            detail=ErrorResponse(error=e.error_code, message=e.message).model_dump())
    except (ValueError, rasterio.errors.RasterioIOError) as e:
        raise HTTPException(status_code=400,
                            detail=ErrorResponse(error="evaluation_failed", message=str(e)).model_dump())

    result = evaluation.evaluate_prediction_truth(rdsm, dem_2d, nodata=nodata, mode=mode)
    predicted_elevation = result.pop("predicted_elevation", None)
    ground_truth = result.pop("ground_truth", None)

    if save_artifacts:
        # The metrics dump mirrors the EXACT response fields reported below;
        # exporting artifacts must never alter them.
        artifact_metrics = {k: v for k, v in result.items()
                            if k in {"mae", "rmse", "correlation",
                                     "correlation_reason", "valid_pixels",
                                     "total_pixels", "held_out_pixel_count",
                                     "calibration_status", "calibration_warning",
                                     "calibration_scale", "calibration_offset",
                                     "raw_correlation_signed", "scale_sign",
                                     "polarity_inverted", "polarity_reason"}}
        try:
            artifacts.export_prediction_artifacts(
                out_root=config.OUTPUT_DIR,
                backend=depth.backend_slug(),
                model_identifier=depth.model_identifier(),
                input_filename=image.filename or "image.png",
                raw=capture.get("raw"),
                relative=rdsm,
                calibrated=predicted_elevation,
                ground_truth=ground_truth,
                metrics=artifact_metrics,
                crs=rgb_meta.get("crs"),
                transform=rgb_meta.get("transform"),
                bounds=rgb_meta.get("bounds"),
            )
        except (ValueError, OSError) as e:
            raise HTTPException(status_code=400,
                                detail=ErrorResponse(error="evaluation_failed",
                                                     message=f"artifact export failed: {e}").model_dump())

    response = EvaluationResponse(**result)

    if include_visualization and predicted_elevation is not None:
        try:
            name = uuid.uuid4().hex
            comparison_path = config.FILES_DIR / name / "comparison.png"
            evaluation.render_height_comparison(rgb, ground_truth, predicted_elevation,
                                                comparison_path)
            response.comparison_image_url = f"/files/{name}/comparison.png"
        except (ValueError, OSError) as e:
            raise HTTPException(status_code=400,
                                detail=ErrorResponse(error="evaluation_failed",
                                                     message=f"visualization failed: {e}").model_dump())

    return response


# -------- static file serving --------------------------------------------- #
@router.get("/files/{job_id}/{filename}")
def file_bundle(job_id: str, filename: str):
    base = config.FILES_DIR / job_id
    # path traversal protection
    name = Path(filename).name
    target = (base / name).resolve()
    if not target.is_file() or not str(target).startswith(str(base.resolve())):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target)
