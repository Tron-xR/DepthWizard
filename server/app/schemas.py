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
    correlation: float
    diff_heatmap_url: str
    # Pixels scored (always the held-out 20% for jobs processed with holdout
    # persistence; null for legacy all-pixel results).
    held_out_pixel_count: Optional[int] = None


class EvaluationResponse(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    correlation: Optional[float] = None
    valid_pixels: int
    total_pixels: int
    held_out_pixel_count: int
    calibrated: bool
    evaluated_units: str
    reason: Optional[str] = None
    comparison_image_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    message: str
