"""Compatibility shim for legacy download_dem imports.

Older scripts and tests expect a top-level module named ``download_dem``.
This repository implements the actual DEM-fetching logic in
``server/app/pipeline/dem_source.py``. This shim re-exports the supported
behaviour so imports keep working while preserving the current code path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np


def download_dem(
    bounds: Iterable[float],
    crs: str = "EPSG:4326",
    *,
    output: Optional[str | Path] = None,
    source: str = "SRTM",
    dem_type: str = "SRTMGL3",
    use_cache: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Download or fetch a DEM tile covering ``bounds``.

    Returns (data_2d, meta_dict) compatible with the project's current
    ``fetch_reference_dem`` implementation.
    """
    from server.app.pipeline.dem_source import fetch_reference_dem

    data, meta = fetch_reference_dem(
        list(bounds),
        crs,
        source=source,
        dem_type=dem_type,
        use_cache=use_cache,
    )

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in {".tif", ".tiff"}:
            import rasterio
            from rasterio.transform import Affine

            transform = meta.get("transform")
            if transform is None:
                transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
            with rasterio.open(
                out,
                "w",
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype="float32",
                crs=meta.get("crs") or crs,
                transform=transform,
                nodata=np.nan,
            ) as dst:
                dst.write(data.astype("float32"), 1)
        else:
            np.save(out, np.asarray(data, dtype="float32"))

    return np.asarray(data, dtype="float32"), meta


# Common aliases used by older scripts/tests.
fetch_reference_dem = lambda bounds, crs="EPSG:4326", **kwargs: download_dem(bounds, crs, **kwargs)[0]
fetch_dem = download_dem
download_dem_for_bounds = download_dem


__all__ = [
    "download_dem",
    "fetch_dem",
    "fetch_reference_dem",
    "download_dem_for_bounds",
]
