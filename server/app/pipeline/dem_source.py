"""Reference DEM fetching for the georeferenced (absolute calibration) branch.

Primary source: OpenTopography (no API key needed for SRTM 30m in "global" DEM).
Also supports a locally-pinned GeoTIFF via DEPTHWIZARD_DEM_FILE for offline tests.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config
from ..errors import DepthWizardError


class DemFetchError(DepthWizardError):
    error_code = "dem_fetch_error"


def _local_dem() -> Optional[np.ndarray]:
    """Return (data, meta) from a locally-configured GeoTIFF if set."""
    path = os.environ.get("DEPTHWIZARD_DEM_FILE")
    if not path:
        return None
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
    return data


# --------------------------------------------------------------------------- #
# Disk cache for network-fetched reference DEMs (separate, reviewable change).
#
# The `/validate` endpoint compares the produced DSM against the same reference
# DEM that calibration fetched, so re-downloading per call wastes bandwidth and
# risks flaky network failures. We cache the fetched tile on disk keyed by
# (source, dem_type, crs, bounds); both calibration and validation share this
# via `fetch_reference_dem`. The local `DEPTHWIZARD_DEM_FILE` override is NOT
# cached (it is already local and env-swappable in tests).
# ponytail: cache entries never expire (SRTM/Copernicus data is static per
# bounds) and the key space is unbounded by unique bboxes; revisit if a mutable
# provider or many distinct queries per deployment appear.
# --------------------------------------------------------------------------- #
def _ref_cache_path(bounds: list, crs: str, source: str, dem_type: str) -> Path:
    import hashlib

    payload = "|".join([source, dem_type, crs or "", repr(list(bounds))])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return config.CACHE_DIR / "ref_dems" / f"{digest}.npz"


def _cache_meta(ref_source: str, crs: str, shape: tuple) -> dict:
    return {
        "source": ref_source,
        "crs": crs,
        "width": shape[1],
        "height": shape[0],
        "transform": None,
        "cached": True,
    }


def fetch_reference_dem(bounds: list, crs: str,
                        source: str = "SRTM",
                        dem_type: str = "SRTMGL3",
                        use_cache: bool = True) -> tuple:
    """Fetch a reference DEM covering `bounds` (in CRS units).

    Returns (data_2d_float32, meta) where meta has 'source', 'transform',
    'width', 'height', 'crs'. Raises DemFetchError on failure.
    """
    # Local override (offline/tests); checked first so an env-swapped DEM_FILE
    # is never shadowed by a stale cached tile.
    local = _local_dem()
    if local is not None:
        # Reuse local array; caller is responsible for resampling/geolocation.
        return local, {
            "source": "local",
            "crs": crs,
            "width": local.shape[1],
            "height": local.shape[0],
            "transform": None,
        }

    if use_cache:
        npz = _ref_cache_path(bounds, crs, source, dem_type)
        if npz.is_file():
            with np.load(npz) as z:
                data = z["data"].astype("float32")
                ref_source = str(z["source"].item()) if "source" in z.files else source
            return data, _cache_meta(ref_source, crs, data.shape)

    data, meta = _fetch_opentopo(bounds, crs, dem_type)

    if use_cache:
        try:
            npz = _ref_cache_path(bounds, crs, source, dem_type)
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, data=data.astype("float32"),
                                source=np.asarray(meta.get("source", source)))
        except Exception:
            # Cache is best-effort: a write failure must never break the fetch.
            pass

    return data, meta


def _fetch_opentopo(bounds: list, crs: str, dem_type: str) -> tuple:
    import requests

    # Convert bounds to lon/lat if CRS is projected, else assume WGS84.
    minx, miny, maxx, maxy = bounds
    if crs and not str(crs).upper().startswith(("EPSG:4326", "OGC:CRS84")):
        west, south, east, north = _to_lonlat(bounds, crs)
    else:
        west, south, east, north = minx, miny, maxx, maxy

    # OpenTopography global DEM API (key is free and recommended)
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
    }
    if config.OPENTOPO_API_KEY:
        params["API_Key"] = config.OPENTOPO_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=config.DEM_FETCH_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as e:
        # Fall back to the no-auth Copernicus GLO-30 AWS Open Data bucket.
        if "401" in str(e) and config.OPENTOPO_API_KEY == "":
            try:
                return _fetch_copernicus_aws(lonmin=west, latmin=south,
                                             lonmax=east, latmax=north)
            except DemFetchError:
                pass
        raise DemFetchError(f"Failed to fetch reference DEM: {e}") from e

    import rasterio
    from rasterio.io import MemoryFile

    try:
        with MemoryFile(resp.content) as mem:
            with mem.open() as src:
                data = src.read(1).astype("float32")
                meta = {
                    "source": dem_type,
                    "crs": str(src.crs),
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "south": south, "north": north, "west": west, "east": east,
                }
        return data, meta
    except Exception as e:
        raise DemFetchError(f"Could not decode reference DEM: {e}") from e


def _to_lonlat(bounds: list, crs: str) -> tuple:
    """Convert projected bounds (CRS) to lon/lat via pyproj (bundled with rasterio)."""
    try:
        import pyproj
    except ImportError as e:  # pragma: no cover
        raise DemFetchError("pyproj not available to convert CRS") from e

    minx, miny, maxx, maxy = bounds
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(minx, miny)
    east, north = transformer.transform(maxx, maxy)
    return west, south, east, north


# --------------------------------------------------------------------------- #
# Copernicus GLO-30 (no-auth AWS Open Data) fallback
# --------------------------------------------------------------------------- #
_BASE_COP = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_prefix(lat: float, lon: float) -> str:
    """Copernicus GLO-30 tile prefix for a coordinate's cell.
    N{n}/S{n} for latitude cell and E{n}/W{n} for longitude cell.
    N{n} covers lat [n, n+1); W{n} covers lon [-n, -n+1).
    Hemisphere letter is derived from the coordinate sign.
    """
    lat_i = int(math.floor(lat))
    ns = "S" if lat < 0 else "N"
    lon_i = int(math.ceil(-lon))
    ew = "W" if lon < 0 else "E"
    return (f"Copernicus_DSM_COG_10_{ns}{abs(lat_i):02d}_00"
            f"_{ew}{abs(lon_i):03d}_00_DEM")


def _fetch_copernicus_aws(lonmin, latmin, lonmax, latmax) -> tuple:
    """Read Copernicus GLO-30 DEM from the public AWS bucket via rasterio
    VSICURL, reading the exact lon/lat bounding-box window from each
    overlapping tile and mosaicking them. No API key needed."""
    import rasterio
    from rasterio.transform import Affine

    if not (-90 <= latmin <= 90 and -180 <= lonmin <= 180 and
            lonmin < lonmax and latmin < latmax):
        raise DemFetchError("Invalid lon/lat bounds for Copernicus fetch")

    # Candidate tile cell indices. Latitude cell = floor(lat).
    # Longitude: tile W{n} covers lon [-n, -n+1), so n = ceil(-lon).
    def lat_cell(lat):
        return int(math.floor(lat))

    def lon_cell(lon):
        return int(math.ceil(-lon))

    lat_lo, lat_hi = lat_cell(latmin), lat_cell(latmax - 1e-9)
    lon_lo_cell, lon_hi_cell = lon_cell(lonmax), lon_cell(lonmin)
    if lon_lo_cell > lon_hi_cell:
        lon_lo_cell, lon_hi_cell = lon_hi_cell, lon_lo_cell

    tiles = []
    lat_sign = -1.0 if latmin < 0 else 1.0
    lon_sign = -1.0 if lonmin < 0 else 1.0
    for la in range(lat_lo, lat_hi + 1):
        for lo in range(lon_lo_cell, lon_hi_cell + 1):
            prefix = _tile_prefix(lat_sign * abs(la), lon_sign * abs(lo))
            url = f"{_BASE_COP}/{prefix}/{prefix}.tif"
            try:
                src = rasterio.open(url)
                # keep if it actually overlaps the requested bbox
                b = src.bounds
                if b.left < lonmax and b.right > lonmin and b.bottom < latmax and b.top > latmin:
                    tiles.append(src)
                else:
                    src.close()
            except Exception:
                continue

    if not tiles:
        raise DemFetchError("No Copernicus GLO-30 tiles found for bounds")

    arrays = []
    for src in tiles:
        try:
            w = src.window(lonmin, latmin, lonmax, latmax)
            if w.width > 0 and w.height > 0:
                # round the window to integer pixel offsets so the read grid
                # has a clean transform rooted at (col0, row0)
                col0 = int(math.floor(w.col_off))
                row0 = int(math.floor(w.row_off))
                col1 = int(math.ceil(w.col_off + w.width))
                row1 = int(math.ceil(w.row_off + w.height))
                arr = src.read(1, window=((row0, row1), (col0, col1)),
                               boundless=True).astype("float32")
                # geographic origin of the (col0,row0) corner of the window
                t = src.transform
                win_tr = Affine(t.a, t.b, t.c + t.a * col0 + t.b * row0,
                                t.d, t.e, t.f + t.d * col0 + t.e * row0)
                # boundless fills the clipped region with nodata/0 -> NaN
                arrays.append((arr, win_tr))
        except Exception:
            continue

    if not arrays:
        for src in tiles:
            src.close()
        raise DemFetchError("Could not read any DEM window from Copernicus tiles")

    out, out_transform = _mosaic(arrays, lonmin, latmin, lonmax, latmax, 1.0 / 3600.0)
    for src in tiles:
        src.close()
    return out, {
        "source": "Copernicus_GLO-30 (AWS)",
        "crs": "EPSG:4326",
        "transform": out_transform,
        "width": out.shape[1],
        "height": out.shape[0],
        "south": latmin, "north": latmax, "west": lonmin, "east": lonmax,
    }


def _mosaic(arrays, lonmin, latmin, lonmax, latmax, res):
    """Build a destination grid over the bbox and paste each tile window in.

    Vectorized via affine index transforms. Returns (array2d, transform).
    """
    import numpy as np
    from rasterio.transform import from_origin

    w = int(round((lonmax - lonmin) / res))
    h = int(round((latmax - latmin) / res))
    dst = np.full((h, w), np.nan, dtype="float32")
    dst_transform = from_origin(lonmin, latmax, res, res)
    inv = ~dst_transform
    for arr, src_transform in arrays:
        rows, cols = arr.shape
        rows_i, cols_i = np.mgrid[0:rows, 0:cols].astype("float64")
        # geographic coords of each pixel
        xs = src_transform.a * cols_i + src_transform.c
        ys = src_transform.e * rows_i + src_transform.f
        # map into destination pixel indices (screen affine: col and row)
        dc = inv.a * xs + inv.c
        dr = inv.e * ys + inv.f
        dc = np.rint(dc).astype("int64")
        dr = np.rint(dr).astype("int64")
        valid = (
            np.isfinite(arr)
            & (dc >= 0) & (dc < w)
            & (dr >= 0) & (dr < h)
        )
        dcol = dc[valid]
        drow = dr[valid]
        vals = arr[valid]
        dst[drow, dcol] = vals
    return dst, dst_transform
