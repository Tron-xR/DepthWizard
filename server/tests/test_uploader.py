"""Unit tests for the upload ingestion + geo-detection logic."""
import numpy as np
import pytest

from app.errors import DepthWizardError
from app.pipeline import uploader
from tests.conftest import write_geotiff, write_rgb_png, make_rgb_gradient


def test_png_detected_as_non_geo(tmp_path):
    p = write_rgb_png(tmp_path / "img.png", make_rgb_gradient())
    arr, meta = uploader.load_raster(p)
    assert not uploader.detect_georeferenced(meta)
    assert arr.shape[0] == 3


def test_geotiff_detected_as_geo(tmp_path):
    elev = np.arange(100, dtype="float32").reshape(10, 10)
    p = write_geotiff(tmp_path / "img.tif", elev)
    arr, meta = uploader.load_raster(p)
    assert uploader.detect_georeferenced(meta)
    assert meta["crs"] == "EPSG:32633"


def test_unsupported_extension_rejected():
    with pytest.raises(DepthWizardError):
        uploader._detect_file_type("video.mp4")


def test_ingest_downsamples_oversized(monkeypatch):
    # Force tiny max dim so our 128px test image triggers downsampling
    from app import config

    monkeypatch.setattr(config, "MAX_WORKING_DIM", 64)
    rgb = make_rgb_gradient(128, 128)
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    info = uploader.ingest_upload(buf.getvalue(), "big.png")
    assert info["width"] == 64 and info["height"] == 64
    assert not info["is_georeferenced"]
