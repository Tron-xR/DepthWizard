"""Pydantic response schemas matching 07-api.md."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    upload_id: str
    is_georeferenced: bool
    crs: Optional[str]
    bounds: Optional[list] = None


class ProcessResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    stage: Optional[str] = None


class ResultResponse(BaseModel):
    heightmap_url: str
    texture_url: str
    dsm_geotiff_url: Optional[str] = None
    min_elev: Optional[float] = None
    max_elev: Optional[float] = None
    cell_size: Optional[float] = None
    world_width: float
    world_depth: float
    is_georeferenced: bool


class ValidationResponse(BaseModel):
    reference_source: str
    landscape_type: Optional[str] = None
    rmse: float
    mae: float
    # None (with a correlation_reason) means the correlation is UNDEFINED -
    # either side has zero variance / too few pixels - not a measured 0.0.
    # 0.0 remains a genuine "zero linear correlation".
    correlation: Optional[float] = None
    correlation_reason: Optional[str] = None
    diff_heatmap_url: str
    # Pixels scored (always the held-out 20% for jobs processed with holdout
    # persistence; null for legacy all-pixel results).
    held_out_pixel_count: Optional[int] = None
    # Flagged when the calibration fit was degenerate (no variance, or a
    # near-zero scale relative to the tile's elevation range, e.g. low relief
    # like a lake). Informational only: the job itself still completes.
    degenerate_calibration: bool = False
    calibration_reason: Optional[str] = None
    # Calibration health of a NON-degenerate fit: "warning" (with a
    # calibration_warning such as "negative_scale" / "prediction_near_constant")
    # means the fit is real but semantically suspicious; scale is reported
    # unchanged (a negative scale is never flipped).
    calibration_status: Optional[str] = None
    calibration_warning: Optional[str] = None
    calibration_scale: Optional[float] = None
    calibration_offset: Optional[float] = None


class EvaluationResponse(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    correlation: Optional[float] = None
    correlation_reason: Optional[str] = None
    valid_pixels: int
    total_pixels: int
    held_out_pixel_count: int
    calibrated: bool
    evaluated_units: str
    reason: Optional[str] = None
    comparison_image_url: Optional[str] = None
    # Same degenerate-calibration flag as /validate; additive, never breaks the
    # shape of an existing response.
    degenerate_calibration: bool = False
    calibration_reason: Optional[str] = None
    # Same calibration health fields as /validate (see ValidationResponse).
    calibration_status: Optional[str] = None
    calibration_warning: Optional[str] = None
    calibration_scale: Optional[float] = None
    calibration_offset: Optional[float] = None


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    message: str
