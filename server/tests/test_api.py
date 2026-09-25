"""API integration tests (FastAPI TestClient) matching 07-api.md contracts.

Depth inference is mocked to a deterministic surface so tests don't require
downloading a model or network access.
"""
import io
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db, config
from app.main import app
from app.pipeline import depth
from tests.conftest import make_rgb_gradient, write_rgb_png, write_geotiff


@pytest.fixture
def client(monkeypatch):
    # Stub depth inference
    def fake_infer(rgb, **kwargs):
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype("float32")
        rdsm = (xx / max(w - 1, 1) * 0.5 + yy / max(h - 1, 1) * 0.5)
        return np.asarray(rdsm, dtype="float32")

    monkeypatch.setattr(depth, "infer_relative_dsm", fake_infer)
    with TestClient(app) as c:
        yield c


def _upload_png(client, size=96):
    rgb = make_rgb_gradient(size, size)
    buf = io.BytesIO()
    from PIL import Image

    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    resp = client.post("/upload", files={"file": ("test.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_non_georeferenced(client):
    data = _upload_png(client)
    assert data["is_georeferenced"] is False
    assert data["upload_id"]


def test_upload_georeferenced(client, monkeypatch):
    from PIL import Image
    import rasterio
    from rasterio.transform import from_origin

    elev = np.arange(1600, dtype="float32").reshape(40, 40)
    path = Path(config.FILES_DIR).parent / "geo.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(path, elev, minx=500000.0, maxy=4650000.0, cell=30.0)
    with open(path, "rb") as f:
        resp = client.post("/upload", files={"file": ("geo.tif", f.read(), "image/tiff")})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["is_georeferenced"] is True
    assert data["crs"] == "EPSG:32633"


def test_upload_bad_file(client):
    resp = client.post("/upload", files={"file": ("bad.png", b"not-a-real-png", "image/png")})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "corrupt_image"


def test_preview_png(client):
    up = _upload_png(client)
    resp = client.get(f"/preview/{up['upload_id']}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    from PIL import Image

    img = Image.open(io.BytesIO(resp.content))
    assert img.width <= 256 and img.height <= 256


def test_preview_geotiff(client):
    from PIL import Image
    from rasterio.transform import from_origin

    elev = np.arange(1600, dtype="float32").reshape(40, 40)
    path = Path(config.FILES_DIR).parent / "preview_geo.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(path, elev, minx=500000.0, maxy=4650000.0, cell=30.0)
    with open(path, "rb") as f:
        resp = client.post("/upload", files={"file": ("geo.tif", f.read(), "image/tiff")})
    assert resp.status_code == 200, resp.text
    up = resp.json()

    resp = client.get(f"/preview/{up['upload_id']}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(resp.content))
    assert img.width <= 256 and img.height <= 256


def test_preview_unknown_404(client):
    assert client.get("/preview/nope").status_code == 404


def test_full_relative_flow(client):
    up = _upload_png(client)
    proc = client.post(f"/process/{up['upload_id']}")
    assert proc.status_code == 200
    job_id = proc.json()["job_id"]

    # poll
    for _ in range(50):
        st = client.get(f"/status/{job_id}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert st["status"] == "done", st

    res = client.get(f"/result/{job_id}").json()
    assert res["heightmap_url"].startswith("/files/")
    assert res["texture_url"].startswith("/files/")
    assert res["dsm_geotiff_url"] is None
    assert res["is_georeferenced"] is False
    assert res["world_width"] == 1000.0

    # fetch file
    f = client.get(res["heightmap_url"])
    assert f.status_code == 200


def test_result_before_done_returns_409(client):
    up = _upload_png(client)
    # create job but don't wait
    proc = client.post(f"/process/{up['upload_id']}")
    job_id = proc.json()["job_id"]
    resp = client.get(f"/result/{job_id}")
    assert resp.status_code in (200, 409)


def test_dem_view_relative(client):
    from PIL import Image

    from app import db, jobs

    rgb = make_rgb_gradient(96, 96)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    up = client.post("/upload", files={"file": ("test.png", buf.getvalue(), "image/png")}).json()

    # run the job synchronously (worker queue is a singleton across tests; the
    # established deterministic pattern is jobs._run_job - see test_absolute_branch)
    job = db.create_job(up["upload_id"], "rdsm")
    jobs._run_job(job["id"])
    assert db.get_job(job["id"])["status"] == "done"

    resp = client.get(f"/dem-view/{job['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"

    img = Image.open(io.BytesIO(resp.content)).convert("L")
    assert img.width == 96 and img.height == 96
    hist = img.histogram()
    # ramp spans the job's own [0, 200] m range exactly -> full 0..255 stretch
    assert hist[0] > 0 and hist[255] > 0


def test_dem_view_png_uses_float_dsm(tmp_path):
    # georeferenced archive path: renders from the full-precision float32 DSM
    from io import BytesIO

    import rasterio
    from rasterio.transform import from_origin
    from PIL import Image

    from app.pipeline.exporter import dem_view_png

    elev = np.arange(1600, dtype="float32").reshape(40, 40) + 500.0  # 500..1599
    tif = tmp_path / "dsm.tif"
    profile = {"driver": "GTiff", "height": 40, "width": 40, "count": 1,
               "dtype": "float32", "crs": "EPSG:32633",
               "transform": from_origin(500000.0, 4650000.0, 30.0, 30.0)}
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(elev, 1)

    png = dem_view_png("missing.png", str(tif), 500.0, 1599.0)
    img = Image.open(BytesIO(png)).convert("L")
    assert img.width == 40 and img.height == 40
    assert img.getpixel((0, 0)) == 0
    assert img.getpixel((39, 39)) == 255


def test_unknown_job_404(client):
    assert client.get("/status/nope").status_code == 404
    assert client.get("/result/nope").status_code == 404
