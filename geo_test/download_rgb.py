"""Fetch matching RGB imagery for the Coloradouge DEM tiles from Esri World
Imagery (no auth), warped/clipped onto each DEM geotiff's grid so pixels align.

Reuses fetch_rgb_esri.py's tile-fetch + reproject-to-DEM approach. Data-gathering
only; not part of the shipped server.

Usage:
    python download_rgb.py
Requires: 'requests', 'rasterio', 'numpy', 'PIL'. Outputs {region}_rgb.tif.
"""
import io
import math
import os

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import reproject

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geo_test_data"))
BASE_TILE = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "World_Imagery/MapServer/tile/{z}/{y}/{x}")
ZOOM = 15
WEB_MERC = 20037508.342789244

REGIONS = {
    "colorado_original": (-105.6, 39.9, -105.4, 40.1),
    "colorado_north":   (-105.6, 40.1, -105.4, 40.3),
    "colorado_west":    (-105.8, 39.9, -105.6, 40.1),
}


def tile_bounds_merc(x, y, z):
    n = 2 ** z
    size = 2 * WEB_MERC / n
    west = -WEB_MERC + x * size
    north = WEB_MERC - y * size
    return west, north, size


def tile_range(bbox, z):
    west, south, east, north = bbox
    n = 2 ** z
    x0 = int((west + 180) / 360 * n)
    x1 = int((east + 180) / 360 * n)
    y0 = int((1 - math.asinh(math.tan(math.radians(north))) / math.pi) / 2 * n)
    y1 = int((1 - math.asinh(math.tan(math.radians(south))) / math.pi) / 2 * n)
    return x0, x1, y0, y1


def fetch_region(bbox, z=ZOOM, name="rgb"):
    x0, x1, y0, y1 = tile_range(bbox, z)
    rows = []
    for yt in range(y0, y1 + 1):
        row = []
        for xt in range(x0, x1 + 1):
            r = requests.get(BASE_TILE.format(z=z, y=yt, x=xt), timeout=60)
            r.raise_for_status()
            row.append(np.array(Image.open(io.BytesIO(r.content)).convert("RGB")))
        rows.append(np.concatenate(row, axis=1))
    mosaic = np.concatenate(rows, axis=0)

    w0, n0, size = tile_bounds_merc(x0, y0, z)
    px = size / 256.0
    src_transform = from_origin(w0, n0, px, px)
    src_crs = "EPSG:3857"

    dem_path = os.path.join(OUT_DIR, f"{name}_dem.tif")
    out_path = os.path.join(OUT_DIR, f"{name}_rgb.tif")
    with rasterio.open(dem_path, "r") as dem:
        dst = np.empty((dem.height, dem.width, 3), dtype="uint8")
        for b in range(3):
            reproject(
                mosaic[:, :, b], dst[:, :, b],
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=dem.transform, dst_crs=dem.crs,
                resampling=rasterio.enums.Resampling.bilinear,
            )
        with rasterio.open(
            out_path, "w", driver="GTiff", height=dem.height,
            width=dem.width, count=3, dtype="uint8",
            transform=dem.transform, crs=dem.crs,
        ) as out:
            out.write(np.transpose(dst, (2, 0, 1)))
    print(f"wrote {out_path} ({dem.height}x{dem.width})")


def main():
    for name, bbox in REGIONS.items():
        fetch_region(bbox, name=name)


if __name__ == "__main__":
    main()
