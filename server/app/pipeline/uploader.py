"""Upload ingestion: validate file type, persist to the file cache, detect
geospatial metadata, and downsample oversized images to a working resolution.

Supported inputs: PNG/JPG (non-georeferenced) and GeoTIFF (georeferenced).
Falls back to the relative branch when a TIFF has no usable CRS/transform.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

import numpy as np

from .. import config
from ..errors import DepthWizardError

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class UnsupportedFileError(DepthWizardError):
    error_code = "unsupported_file_type"


class CorruptImageError(DepthWizardError):
    error_code = "corrupt_image"


def _detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".tif", ".tiff"):
        return "tiff"
    if ext in (".png",):
        return "png"
    if ext in (".jpg", ".jpeg"):
        return "jpg"
    raise UnsupportedFileError(f"Only PNG, JPG, and GeoTIFF are supported. Got: {filename}")


def load_raster(path: Path):
    """Load a raster, returning (data[bands,H,W] float32, meta dict).
    Uses rasterio for TIFF (preserves CRS/transform) and Pillow for PNG/JPG.
    """
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        import rasterio

        try:
            with rasterio.open(path) as src:
                data = src.read().astype("float32")
                meta = {
                    "crs": str(src.crs) if src.crs else None,
                    "transform": src.transform,
                    "bounds": list(src.bounds),
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                }
            return data, meta
        except Exception as e:
            raise CorruptImageError(f"Could not read GeoTIFF: {e}") from e

    # PNG / JPG
    try:
        img = Image.open(path)
        img.load()
        rgb = img.convert("RGB")
    except Exception as e:
        raise CorruptImageError(f"Could not read image: {e}") from e
    import numpy as np

    arr = np.asarray(rgb, dtype="float32").transpose(2, 0, 1)
    meta = {
        "crs": None,
        "transform": None,
        "bounds": None,
        "width": arr.shape[2],
        "height": arr.shape[1],
        "count": arr.shape[0],
    }
    return arr, meta


def detect_georeferenced(meta: dict) -> bool:
    """Georeferenced only if we have both a CRS and a transform/bounds."""
    return bool(meta.get("crs") and (meta.get("transform") is not None or meta.get("bounds")))


def _downsample_array(arr, max_dim: int) -> tuple:
    """Downsample so the largest dimension <= max_dim. Returns (array, scale)."""
    import numpy as np

    h, w = arr.shape[1], arr.shape[2]
    largest = max(h, w)
    if largest <= max_dim:
        return arr, 1.0

    scale = max_dim / largest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr.transpose(1, 2, 0).astype("uint8") if arr.shape[0] == 3
                          else arr[0].astype("uint8"), "RGB" if arr.shape[0] == 3 else "L")
    img = img.resize((new_w, new_h), Image.BILINEAR)
    out = np.asarray(img, dtype="float32")
    if arr.shape[0] == 3:
        out = out.transpose(2, 0, 1)
    else:
        out = out[None, :, :]
    return out, scale


def _preview_rgb(arr) -> "np.ndarray":
    """float[bands,H,W] -> uint8[H,W,3] for a cosmetic thumbnail."""
    import numpy as np

    if arr.ndim == 3 and arr.shape[0] >= 3:
        img = arr[:3].transpose(1, 2, 0)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        img = np.stack([arr[0]] * 3, axis=-1)
    else:
        img = arr.transpose(1, 2, 0) if arr.ndim == 3 else arr

    lo, hi = float(img.min()), float(img.max())
    out = ((img - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(img)
    return np.clip(out, 0, 255).astype("uint8")


def render_preview_png(path: Path, max_dim: int = 256) -> bytes:
    """Render a small PNG thumbnail of the cached working raster.

    Purely cosmetic (Unity file-picker preview): reads back the already
    downsampled working file via load_raster and 8-bit normalizes it. Never
    used by the pipeline or metrics.
    """
    import io

    arr, _ = load_raster(path)
    img = Image.fromarray(_preview_rgb(arr), "RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def ingest_upload(file_bytes: bytes, filename: str) -> dict:
    """Persist an uploaded file into the cache and return metadata.

    Returns dict: file_path, file_type, is_georeferenced, crs, bounds, height, width.
    """
    file_type = _detect_file_type(filename)
    ext_map = {"tiff": ".tif", "png": ".png", "jpg": ".jpg"}
    dest = config.CACHE_DIR / f"{uuid.uuid4().hex}{ext_map[file_type]}"
    dest.write_bytes(file_bytes)

    arr, meta = load_raster(dest)
    is_geo = detect_georeferenced(meta)
    crs = meta.get("crs") if is_geo else None
    notes = "as-is" if arr.shape[1] <= config.MAX_WORKING_DIM else "downsampled"

    # Downsample oversized images to the working resolution
    working, _ = _downsample_array(arr, config.MAX_WORKING_DIM)
    _write_working(dest, working, meta, is_geo, file_type)

    return {
        "file_path": str(dest),
        "file_type": file_type,
        "is_georeferenced": is_geo,
        "crs": crs,
        "bounds": meta.get("bounds"),
        "height": int(working.shape[1]),
        "width": int(working.shape[2]),
        "resample_note": notes,
    }


def _write_working(path: Path, arr, meta: dict, is_geo: bool, file_type: str) -> None:
    """Overwrite the cached file with the (possibly downsampled) working raster."""
    import numpy as np

    h, w = arr.shape[1], arr.shape[2]
    if file_type == "tiff":
        import rasterio
        from rasterio.transform import from_origin

        transform = meta.get("transform") if (is_geo and meta.get("transform") is not None) \
            else from_origin(0, h, 1, 1)
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": arr.shape[0],
            "dtype": "float32",
            "crs": meta.get("crs") if is_geo else None,
            "transform": transform,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr)
        return

    # PNG / JPG
    if arr.shape[0] == 3:
        img = arr.transpose(1, 2, 0)
    else:
        img = arr[0]
    lo, hi = float(img.min()), float(img.max())
    img8 = np.clip(((img - lo) / (hi - lo) * 255.0) if hi > lo else (img * 0.0), 0, 255).astype("uint8")
    pil = Image.fromarray(img8, "RGB" if arr.shape[0] == 3 else "L")
    if file_type == "jpg":
        pil.convert("RGB").save(path, "JPEG", quality=95)
    else:
        pil.save(path, "PNG")
