"""Export of pipeline outputs: unified heightmap (PNG/EXR for Unity), source
texture, and georeferenced DSM GeoTIFF. Also computes world dimensions for the
Unity mesh generator from cell size / bounds.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


def _normalize_8bit(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo == 0:
        return np.zeros_like(arr, dtype="uint8")
    return ((arr - lo) / (hi - lo) * 255).astype("uint8")


def dem_view_png(heightmap_path: str, dsm_geotiff_path: Optional[str],
                 min_elev: Optional[float], max_elev: Optional[float]) -> bytes:
    """Grayscale DEM overlay PNG of the true-scale elevation grid.

    Renders the same elevation array the Unity mesh consumed, min-max
    normalized to this job's own range, never exaggerated. Uses the
    full-precision float32 dsm.tif when a georeferenced job archived one;
    otherwise de-normalizes the 8-bit heightmap back to meters with the stored
    min/max (the exact array that became mesh vertices). Cosmetic only - like
    /preview it never feeds the pipeline.
    """
    from io import BytesIO

    from PIL import Image

    if dsm_geotiff_path and Path(dsm_geotiff_path).is_file():
        import rasterio

        with rasterio.open(dsm_geotiff_path) as src:
            elev = src.read(1).astype("float64")
    else:
        arr8 = np.asarray(Image.open(heightmap_path).convert("L")).astype("float64")
        lo, hi = float(min_elev or 0.0), float(max_elev or 0.0)
        if hi <= lo:
            hi = lo + 1.0
        elev = arr8 / 255.0 * (hi - lo) + lo
    buf = BytesIO()
    Image.fromarray(_normalize_8bit(elev), "L").save(buf, "PNG")
    return buf.getvalue()


def export_heightmap(elevation: np.ndarray, path: Path) -> dict:
    """Write an 8-bit grayscale heightmap PNG for the Unity mesh generator.

    Returns min/max elevation so Unity can de-normalize.
    ponytail: 8-bit quantization (~1% of range) is fine for visualization; a
    16-bit I;16 PNG is unreliable in Unity's runtime DownloadHandlerTexture.
    Revisit if a sub-meter absolute DSM ever needs full precision.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr8 = _normalize_8bit(elevation)
    Image.fromarray(arr8, "L").save(path, "PNG")
    return {"min_elev": float(elevation.min()), "max_elev": float(elevation.max())}


def export_texture(rgb: np.ndarray, path: Path) -> None:
    """Write the source RGB as an 8-bit texture for the mesh. rgb is H,W,3."""
    from PIL import Image

    rgb8 = np.clip(rgb, 0, 255).astype("uint8") if rgb.max() <= 255 else \
        (np.clip(rgb, 0.0, 1.0) * 255).astype("uint8")
    Image.fromarray(rgb8, "RGB").save(path, "PNG")


def export_geotiff(elevation: np.ndarray, crs: str, transform, bounds, path: Path) -> None:
    """Write a georeferenced DSM GeoTIFF with correct CRS/transform."""
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = transform if transform is not None else from_origin(0, elevation.shape[0], 1, 1)
    profile = {
        "driver": "GTiff",
        "height": elevation.shape[0],
        "width": elevation.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(elevation.astype("float32"), 1)


def export_dsm_geotiff(source_tif: Path, out_path: Path, *, job_id: str,
                       input_filename: str, backend: str, model: str,
                       min_elev: Optional[float], max_elev: Optional[float]) -> str:
    """Write a downloadable, traceability-tagged copy of a job's real DSM.

    Re-uses the pipeline-written float32 GeoTIFF (its CRS/transform come from
    the original upload meta in jobs._calibrate_absolute), so pixel values stay
    byte-identical to the mesh's true-scale heights; the copy only adds deflate
    + GDAL metadata tags naming this as a MODEL-ESTIMATED surface (no vertical
    exaggeration). Returns the output path.
    """
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_tif) as src:
        data = src.read(1)
        profile = src.profile.copy()
    profile.update(compress="deflate", nodata=None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            SOURCE="DepthWizard model-estimated DSM (NOT ground truth)",
            MODEL=model,
            BACKEND=backend,
            JOB_ID=job_id,
            INPUT=Path(input_filename).name,
            VERTICAL_EXAGGERATION="none",
            MIN_ELEVATION_M=f"{min_elev:.6g}" if min_elev is not None else "",
            MAX_ELEVATION_M=f"{max_elev:.6g}" if max_elev is not None else "",
        )
    return str(out_path)


def compute_world_dimensions(height: int, width: int,
                             cell_size: Optional[float],
                             bounds: Optional[list] = None,
                             crs: Optional[str] = None) -> tuple:
    """Return (world_width, world_depth) in meters for Unity mesh scaling.

    Georeferenced (cell_size set): scale the raster to real meters.
    - Geographic CRS (e.g. EPSG:4326): bounds are degrees; convert each span
      to meters using 111320 m/deg lat and 111320*cos(lat) m/deg lon so a
      30N tile does not render as a collapsed near-1D strip.
    - Projected CRS: pixel size is already metric, so width*cell_size works.
    Otherwise fall back to a legible 1000-unit square (relative mode).
    """
    if cell_size and cell_size > 0:
        if bounds and crs and _is_geographic(crs):
            minx, miny, maxx, maxy = bounds
            center_lat = math.radians((miny + maxy) / 2.0)
            return (float((maxx - minx) * 111320.0 * math.cos(center_lat)),
                    float((maxy - miny) * 111320.0))
        return float(width * cell_size), float(height * cell_size)
    return 1000.0, 1000.0


def _is_geographic(crs: str) -> bool:
    from rasterio.crs import CRS

    c = CRS.from_user_input(crs)
    return bool(c) and c.is_geographic
