"""Unit tests for the reference-DEM disk cache (historically re-downloaded)."""
import numpy as np

from app.pipeline import dem_source
from app.pipeline.dem_source import _ref_cache_path
from tests.conftest import write_geotiff

_BOUNDS = [1.0, 2.0, 3.0, 4.0]
_CRS = "EPSG:4326"


def _fake_fetch(data, calls):
    def fake(bounds, crs, dem_type):
        calls["n"] += 1
        return data, {"source": "SRTMGL3", "crs": crs,
                      "width": data.shape[1], "height": data.shape[0],
                      "transform": None}
    return fake


def test_cached_fetch_hits_disk_not_network(monkeypatch):
    calls = {"n": 0}
    data = np.arange(64, dtype="float32").reshape(8, 8)
    monkeypatch.setattr(dem_source, "_fetch_opentopo", _fake_fetch(data, calls))

    a, m1 = dem_source.fetch_reference_dem(_BOUNDS, _CRS)
    b, m2 = dem_source.fetch_reference_dem(_BOUNDS, _CRS)

    assert calls["n"] == 1, "second fetch must reuse the disk cache"
    np.testing.assert_array_equal(a, b)
    assert m1["source"] == "SRTMGL3"
    assert m2["source"] == "SRTMGL3"
    assert m2["cached"] is True
    assert _ref_cache_path(_BOUNDS, _CRS, "SRTM", "SRTMGL3").is_file()


def test_fetch_with_cache_disabled_refetches(monkeypatch):
    calls = {"n": 0}
    data = np.zeros((4, 4), dtype="float32")
    monkeypatch.setattr(dem_source, "_fetch_opentopo", _fake_fetch(data, calls))

    dem_source.fetch_reference_dem(_BOUNDS, _CRS, use_cache=False)
    dem_source.fetch_reference_dem(_BOUNDS, _CRS, use_cache=False)

    assert calls["n"] == 2
    # nothing cached for this key when use_cache=False
    assert not _ref_cache_path(_BOUNDS, _CRS, "SRTM", "SRTMGL3").is_file()


def test_local_override_bypasses_cache(monkeypatch):
    """The DEPTHWIZARD_DEM_FILE override must bypass the cache entirely."""
    calls = {"n": 0}
    data = np.ones((4, 4), dtype="float32")
    monkeypatch.setattr(dem_source, "_fetch_opentopo", _fake_fetch(data, calls))
    monkeypatch.setattr(dem_source, "_local_dem", lambda: data)

    out, meta = dem_source.fetch_reference_dem(_BOUNDS, _CRS)

    assert calls["n"] == 0
    assert meta["source"] == "local"
    assert not _ref_cache_path(_BOUNDS, _CRS, "SRTM", "SRTMGL3").is_file()


def test_calibration_fetch_uses_dedicated_source_only(monkeypatch, tmp_path):
    """Calibration may pin its own DEM (DEPTHWIZARD_CALIBRATION_DEM_FILE) so
    Copernicus GLO-30 stays a validation-only reference; the shared fetch
    (validation) is untouched.

    Non-trivial logic (a second local-override path that must NOT be shadowed by
    / re-use the validation fetch): ONE runnable check - a real GeoTIFF drives
    the calibration fetch while the validation fetch ignores it.
    """
    calls = {"n": 0}
    data = np.arange(16, dtype="float32").reshape(4, 4)
    monkeypatch.setattr(dem_source, "_fetch_opentopo", _fake_fetch(data, calls))
    calib_tif = tmp_path / "calib_ref.tif"
    write_geotiff(calib_tif, data + 100.0)
    monkeypatch.setenv("DEPTHWIZARD_CALIBRATION_DEM_FILE", str(calib_tif))

    cal_out, cal_meta = dem_source.fetch_calibration_reference_dem(_BOUNDS, _CRS)
    # calibration hit the dedicated file (its own values + source), touching
    # neither the network nor the shared ref_dems cache...
    np.testing.assert_array_equal(cal_out, data + 100.0)
    assert cal_meta["source"] == "local"
    assert calls["n"] == 0
    assert not _ref_cache_path(_BOUNDS, _CRS, "SRTM", "SRTMGL3").is_file()

    # ...the validation fetch still went to the shared path and saw no override.
    ref_out, ref_meta = dem_source.fetch_reference_dem(_BOUNDS, _CRS)
    assert calls["n"] == 1
    assert ref_meta["source"] == "SRTMGL3"


def test_calibration_fetch_falls_back_to_shared_without_override(monkeypatch, tmp_path):
    """Without DEPTHWIZARD_CALIBRATION_DEM_FILE the calibration fetch must
    behave exactly like fetch_reference_dem (offline tests stay green)."""
    calls = {"n": 0}
    data = np.zeros((4, 4), dtype="float32")
    monkeypatch.setattr(dem_source, "_fetch_opentopo", _fake_fetch(data, calls))

    out, meta = dem_source.fetch_calibration_reference_dem(_BOUNDS, _CRS)

    assert calls["n"] == 1
    assert meta["source"] == "SRTMGL3"
    np.testing.assert_array_equal(out, data)