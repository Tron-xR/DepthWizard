"""Progressive Morphological Filter (Zhang et al., 2003) on the height raster.

Correctness checks the tide is meant to catch:
  (a) a DSM with sharp rectangular "buildings" and a scattered-noise "tree
      canopy" on a smooth known terrain base must come OUT with the objects
      substantially removed, and
  (b) the smooth terrain base shape must be LARGELY PRESERVED - the filtered
      output is compared against the KNOWN ground-truth base, not against the
      spiky input, because a filter that merely smears the input would reduce
      (a) while failing (b).
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.pipeline import dem_filter
from tests.conftest import write_geotiff


def _synthetic_dsm(h=128, w=128, cell=1.0, seed=7):
    """Known terrain base + sharp building spikes + scattered tree canopy."""
    yy, xx = np.mgrid[0:h, 0:w].astype("float64")
    base = (150.0 + 6.0 * np.sin(xx / w * 2 * np.pi)
            + 4.0 * np.cos(yy / h * 2 * np.pi))
    spiky = base.copy()
    spiky[30:46, 20:36] += 20.0  # 16x16 m building
    spiky[70:82, 60:72] += 15.0  # 12x12 m building
    rng = np.random.default_rng(seed)
    canopy = (slice(20, 90), slice(80, 120))
    noise = np.zeros_like(base)
    noise[canopy] = rng.uniform(2.0, 8.0, noise[canopy].shape)
    noise[rng.uniform(size=noise.shape) < 0.85] = 0.0
    spiky += noise
    return base, spiky, noise


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_progressive_morphological_filter_removes_objects_preserves_terrain():
    base, spiky, noise = _synthetic_dsm()
    dem = dem_filter.progressive_morphological_filter(spiky, 1.0)

    building = np.zeros_like(base, dtype=bool)
    building[30:46, 20:36] = True
    building[70:82, 60:72] = True
    canopy = noise > 0

    # (a) buildings: the input sat 18+ m above the terrain base at these pixels;
    # the filtered output must land back on the base, not keep the spike.
    dsm_excess = float((spiky[building] - base[building]).mean())
    dem_excess = float((dem[building] - base[building]).mean())
    assert dsm_excess > 15.0  # the synthetic spikes are real and tall
    assert dem_excess < 1.0, f"buildings not removed: still {dem_excess:.2f} m up"
    assert (spiky[building] - dem[building]).mean() > 15.0  # reduction

    # (a) tree canopy: the scattered bumps must be pulled down to the base.
    dsm_canopy_excess = float((spiky[canopy] - base[canopy]).mean())
    dem_canopy_excess = float((dem[canopy] - base[canopy]).mean())
    assert dsm_canopy_excess > 3.0
    assert dem_canopy_excess < 1.0, f"canopy not removed: still {dem_canopy_excess:.2f} m up"
    assert (spiky[canopy] - dem[canopy]).mean() > 3.0  # reduction

    # (b) terrain shape preservation vs the KNOWN base: the filtered surface must
    # track the base far more closely than the spiky input did, and must be
    # essentially exact on open ground (never a smoothed-over mash).
    assert _rmse(spiky, base) > 1.0            # input really is spiky
    assert _rmse(dem, base) < 1.0, f"DEM drifted from base: RMSE {_rmse(dem, base):.3f}"
    open_ground = ~(building | canopy)
    assert _rmse(dem[open_ground], base[open_ground]) < 0.01


def test_filter_leaves_flat_and_linear_terrain_untouched():
    # A flat trial and a monotone ramp must pass through bit-identical: the
    # opening of a smooth/linear surface equals the surface, so every diff is 0.
    flat = np.full((64, 64), 100.0)
    flat[30, 30] = 105.0  # single spike must be removed, though
    filtered_flat = dem_filter.progressive_morphological_filter(flat, 1.0)
    assert np.abs(filtered_flat - 100.0).max() <= 0.0

    ramp = 100.0 + 0.5 * np.arange(64)[:, None]
    filtered_ramp = dem_filter.progressive_morphological_filter(ramp, 1.0)
    assert np.array_equal(filtered_ramp, ramp.astype("float32"))


def test_export_dem_route_filters_building(tmp_path):
    """/export-dem/{job_id} reuses the DSM's raster/CRS/transform and comes out
    measurably lower exactly where the building stood, matching the DSM
    elsewhere - i.e. a genuinely separate export, not a relabeled DSM copy."""
    from rasterio.transform import from_origin

    # synthetic job raster: gentle slope ground + 8x8 m building +12 m
    yy, xx = np.mgrid[0:64, 0:64].astype("float64")
    base = 150.0 + 0.2 * xx
    dsm = base.copy()
    dsm[20:28, 20:28] += 12.0
    tif = tmp_path / "dsm.tif"
    write_geotiff(tif, dsm, minx=500000.0, maxy=4650064.0, cell=1.0)

    upload = db.create_upload(
        original_filename="synth.tif",
        file_path=str(tif),
        file_type="tiff",
        is_georeferenced=True,
        crs="EPSG:32633",
        bounds=[500000.0, 4650000.0, 500064.0, 4650064.0],
    )
    job = db.create_job(upload["id"], "absolute_dsm")
    db.create_result(
        job_id=job["id"],
        heightmap_path=str(tmp_path / "heightmap.png"),
        texture_path=str(tmp_path / "texture.png"),
        dsm_geotiff_path=str(tif),
        min_elev=float(dsm.min()),
        max_elev=float(dsm.max()),
        cell_size=1.0,
        world_width=64.0,
        world_depth=64.0,
    )
    db.set_job_status(job["id"], "done")

    with TestClient(app) as c:
        resp = c.get(f"/export-dem/{job['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/tiff"

    import rasterio

    out = tmp_path / "dem_export.tif"
    out.write_bytes(resp.content)
    with rasterio.open(out) as src:
        dem = src.read(1).astype("float64")
        assert src.crs.to_string() == "EPSG:32633"
        assert src.transform == from_origin(500000.0, 4650064.0, 1.0, 1.0)
        assert "bare-earth" in src.tags()["SOURCE"].lower()
        assert "morphological" in src.tags()["FILTER"].lower()

    building = np.zeros_like(base, dtype=bool)
    building[20:28, 20:28] = True
    open_ground = ~building
    # removal at the building (DSM sits 12 m high, DEM must be back on the base)
    dsm_excess = float((dsm[building] - base[building]).mean())
    dem_excess = float((dem[building] - base[building]).mean())
    assert dsm_excess > 10.0
    assert dem_excess < 2.0, f"building not removed: still {dem_excess:.2f} m up"
    assert float((dsm[building] - dem[building]).mean()) > 8.0
    # open, flat-slopping ground preserved essentially exactly
    assert np.abs(dem[open_ground] - base[open_ground]).max() <= 0.02
    # measurably different from the DSM export -> not a relabeled copy
    assert not np.array_equal(dem, dsm)


def test_export_dem_requires_georeferenced(tmp_path):
    """Relative-only jobs have no CRS or real scale: same clear 4xx as DSM."""
    upload = db.create_upload(
        original_filename="rel.png",
        file_path=str(tmp_path / "nope.png"),
        file_type="png",
        is_georeferenced=False,
    )
    job = db.create_job(upload["id"], "rdsm")
    db.create_result(
        job_id=job["id"],
        heightmap_path=str(tmp_path / "heightmap.png"),
        texture_path=str(tmp_path / "texture.png"),
    )
    db.set_job_status(job["id"], "done")

    with TestClient(app) as c:
        resp = c.get(f"/export-dem/{job['id']}")
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "dsm_export_requires_georeferenced"


@pytest.mark.parametrize("bad_cell", [0.0, -1.0])
def test_filter_rejects_non_positive_cell(bad_cell):
    with pytest.raises(ValueError):
        dem_filter.progressive_morphological_filter(np.zeros((8, 8)), bad_cell)
