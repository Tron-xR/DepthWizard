"""Shared test fixtures: sample images, DB isolation, and a demo upload."""
from pathlib import Path

import numpy as np
import pytest

import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from app import config, db  # noqa: E402

TEST_DATA = SERVER_ROOT / "tests" / "_data"


def make_rgb_gradient(h=128, w=128) -> np.ndarray:
    """A synthetic color image: a smooth bright-to-dark gradient with a blob,
    giving the depth model something to work with."""
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    base = (xx / max(w - 1, 1))[:, :, None]
    g = np.concatenate(
        [(base * 255), (0.5 * 255 * np.ones_like(base)), ((1 - base) * 255)], axis=2
    )
    # a circular "hill" brighter spot
    cy, cx = h // 2, w // 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    blob = np.clip(60 * np.exp(-(dist / (w / 6)) ** 2), 0, 255)[:, :, None]
    g = np.clip(g + blob, 0, 255)
    return g.astype("uint8")


def write_rgb_png(path: Path, rgb: np.ndarray) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(path, "PNG")
    return path


def write_geotiff(path: Path, elevation: np.ndarray, crs: str = "EPSG:32633",
                  minx: float = 500000.0, maxy: float = 4650000.0,
                  cell: float = 30.0) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(minx, maxy, cell, cell)
    profile = {
        "driver": "GTiff", "height": elevation.shape[0], "width": elevation.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(elevation.astype("float32"), 1)
    return path


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Give each test its own data directory + initialized DB."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "CACHE_DIR", data_dir / "cache")
    monkeypatch.setattr(config, "FILES_DIR", data_dir / "files")
    monkeypatch.setattr(config, "DB_PATH", data_dir / "depthwizard.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", data_dir / "outputs")
    config.ensure_dirs()
    db.init_db()
    yield
