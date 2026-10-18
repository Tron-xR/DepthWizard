"""HR dataset build tests: site design, split freezing, tile products.

Everything here is synthetic and offline: fake STAC pages, hand-written
GeoTIFFs, a tmp download cache. The commits' frozen test sites are pinned so
regenerating the site table with a different seed fails loudly.
"""

import json

import numpy as np
import pytest
from rasterio.transform import from_origin

from app.pipeline import evaluation, hr_dataset, hr_download

FROZEN_TEST_SITES = {
    "aegeri", "arolla", "chasseral", "geneva_center",
    "lausanne_center", "napf", "saas_fee", "sihlwald",
}

CELL = "2600-1200"
TILE_PX = 32


def _site(**overrides):
    row = {"site_id": "testsite", "stratum": "urban", "split": "test",
           "lon_wgs84": 7.44, "lat_wgs84": 46.95,
           "e_lv95": 2600512, "n_lv95": 1200512, "rationale": "synthetic"}
    row.update(overrides)
    return row


def _tile():
    return hr_dataset.site_tiles(_site(), gsd_m=1.0, tile_px=TILE_PX)[3]


def _write_tif(path, array, transform, crs="EPSG:2056", nodata=np.nan):
    import rasterio

    array = np.asarray(array, dtype="float32")
    count = 1 if array.ndim == 2 else array.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[-2],
                       width=array.shape[-1], count=count, dtype="float32",
                       crs=crs, transform=transform, nodata=nodata) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            dst.write(array, indexes=list(range(1, count + 1)))


def _read_tif(path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.shape, str(src.crs)


def _synthetic_downloads(root, *, rgb_width=60):
    """dsm=100.7 with a 98.0 patch, dtm=100.0, coarse=98.0, RGB bands."""
    dsm_name = f"swisssurface3d-raster_2021_{CELL}_0.5_2056_5728.tif"
    dtm_name = f"swissalti3d_2019_{CELL}_0.5_2056_5728.tif"
    rgb_name = f"swissimage-dop10_2019_{CELL}_0.1_2056.tif"

    _write_tif(root / "hr_downloads" / "rgb" / rgb_name,
               np.stack([np.full((60, rgb_width), v, dtype="float32")
                         for v in (10.0, 20.0, 30.0)]),
               from_origin(2600500, 1200530, 1, 1))
    dsm = np.full((120, 120), 100.7, dtype="float32")
    dsm[60:88, 60:88] = 98.0
    _write_tif(root / "hr_downloads" / "dsm" / dsm_name, dsm,
               from_origin(2600500, 1200530, 0.5, 0.5))
    _write_tif(root / "hr_downloads" / "dtm" / dtm_name,
               np.full((120, 120), 100.0, dtype="float32"),
               from_origin(2600500, 1200530, 0.5, 0.5))

    transform = _tile()["transform"]
    minx, miny, maxx, maxy = hr_dataset.tile_bounds_wgs84(transform, TILE_PX,
                                                          TILE_PX)
    res = 0.0005
    minx, miny = minx - 0.005, miny - 0.005
    maxx, maxy = maxx + 0.005, maxy + 0.005
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    _write_tif(root / "hr_downloads" / "cop30"
               / "Copernicus_DSM_COG_10_N46_00_E007_00_DEM.tif",
               np.full((height, width), 98.0, dtype="float32"),
               from_origin(minx, maxy, res, res), crs="EPSG:4326")
    return {"rgb": rgb_name, "dsm": dsm_name, "dtm": dtm_name}


def _checksum_index(tmp_path, files):
    index = tmp_path / "hr_downloads.csv"
    records = [
        {"path": f"hr_downloads/{product}/{name}", "year": year}
        for (product, name), year in (
            (("rgb", files["rgb"]), "2019"),
            (("dsm", files["dsm"]), "2021"),
            (("dtm", files["dtm"]), "2019"),
        )
    ]
    hr_download.save_checksums(index, records)
    return index


def _water_box():
    return {"type": "Polygon", "coordinates": [[
        (2600520, 1200490), (2600530, 1200490), (2600530, 1200500),
        (2600520, 1200500), (2600520, 1200490)]]}


# --------------------------------------------------------------------------- #
# Site table + splits
# --------------------------------------------------------------------------- #

def test_site_table_balanced_and_valid():
    rows = hr_dataset.build_site_rows()
    assert len(rows) == 60
    assert hr_dataset.validate_sites(rows) == []
    for stratum in hr_dataset.STRATA:
        splits = [r["split"] for r in rows if r["stratum"] == stratum]
        expected = ([s for s, count in hr_dataset.SPLIT_SIZES.items()
                     for _ in range(count)]
                    + ["val"] * hr_dataset.EXTRA_VAL_PER_STRATUM
                    + ["train"] * (hr_dataset.EXTRA_SITES_PER_STRATUM
                                   - hr_dataset.EXTRA_VAL_PER_STRATUM))
        assert sorted(splits) == sorted(expected)


def test_frozen_test_sites():
    rows = hr_dataset.build_site_rows()
    test_sites = {r["site_id"] for r in rows if r["split"] == "test"}
    assert test_sites == FROZEN_TEST_SITES


def test_extension_sites_are_train_or_val_only():
    rows = hr_dataset.build_site_rows()
    extension = [r for r in rows
                 if r["site_id"] not in hr_dataset.FROZEN_SPLITS]
    assert len(extension) == 32
    assert {r["split"] for r in extension} == {"val", "train"}


def test_committed_site_table_matches_seed():
    committed = {r["site_id"]: r for r in hr_dataset.load_sites_csv()}
    built = {r["site_id"]: r for r in hr_dataset.build_site_rows()}
    assert committed.keys() == built.keys()
    for site_id, row in built.items():
        assert committed[site_id]["stratum"] == row["stratum"]
        assert committed[site_id]["split"] == row["split"]


def test_assign_splits_is_deterministic():
    sites = hr_dataset.build_site_rows(seed=0)
    again = hr_dataset.assign_splits(
        [{"site_id": r["site_id"], "stratum": r["stratum"]} for r in sites],
        seed=hr_dataset.SEED)
    first = hr_dataset.assign_splits(
        [{"site_id": r["site_id"], "stratum": r["stratum"]} for r in sites],
        seed=hr_dataset.SEED)
    assert first == again


def test_validate_sites_rejects_close_pair():
    row_a = _site(site_id="a")
    row_b = _site(site_id="b", e_lv95=2600512 + 1000)
    rows = [row_a, row_b] + [
        _site(site_id=f"pad{i}", e_lv95=2600512 + 3000 * (i + 1))
        for i in range(26)]
    errors = hr_dataset.validate_sites(rows)
    assert any("a" in e and "b" in e and "km apart" in e for e in errors)


# --------------------------------------------------------------------------- #
# Tile layout + STAC planning (offline)
# --------------------------------------------------------------------------- #

def test_site_tiles_layout_and_alignment():
    tiles = hr_dataset.site_tiles(_site(), gsd_m=1.0, tile_px=TILE_PX)
    assert [t["tile_id"] for t in tiles] == [
        "testsite_00", "testsite_01", "testsite_10", "testsite_11"]
    for tile in tiles:
        assert tile["width"] == tile["height"] == TILE_PX
        assert tile["transform"].a == 1.0
        assert tile["transform"].e == -1.0
        assert tile["split"] == "test"
    first, last = tiles[0], tiles[3]
    assert first["transform"].c == 2600512 - TILE_PX
    assert first["transform"].f == 1200512 + TILE_PX  # top row of the 2x2 grid
    assert last["transform"].c == 2600512
    assert last["transform"].f == 1200512


def test_sheet_cells():
    assert hr_dataset.sheet_cells(2600512, 1200480, 2600544, 1200512) == [CELL]
    cells = hr_dataset.sheet_cells(2599500, 1199500, 2600500, 1200500)
    assert sorted(cells) == ["2599-1199", "2599-1200", "2600-1199", "2600-1200"]


def test_choose_items_prefers_newest_epoch():
    features = [
        {"id": f"swissimage-dop10_{year}_{CELL}_0.1_2056"}
        for year in ("2019", "2022", "2017")
    ]
    features.append({"id": "swissimage-dop10_2022_2601-1200_0.1_2056"})
    chosen = hr_dataset.choose_items(features, {CELL})
    assert chosen[CELL]["id"].startswith("swissimage-dop10_2022")
    assert hr_dataset.item_year(chosen[CELL]) == "2022"


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self, n=-1):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_plan_site_downloads_parses_fake_stac():
    def feature(collection, year, cell, asset):
        return {"id": f"{collection}_{year}_{cell}",
                "assets": {"default": {"href": f"https://srv/{collection}_"
                                               f"{year}_{cell}{asset}"}}}

    pages = {
        "swissimage-dop10": [
            feature("swissimage-dop10", "2019", CELL, "_0.1_2056.tif"),
            feature("swissimage-dop10", "2022", CELL, "_0.1_2056.tif")],
        "swisssurface3d-raster": [
            feature("swisssurface3d-raster", "2021", CELL,
                    "_0.5_2056_5728.tif")],
        "swissalti3d": [
            feature("swissalti3d", "2020", CELL, "_0.5_2056_5728.tif")],
    }

    def fake_fetch(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        collection = url.split("/collections/")[1].split("/items")[0]
        collection = collection.removeprefix("ch.swisstopo.")
        return _FakeResponse(json.dumps(
            {"features": pages.get(collection, []), "links": []}).encode())

    requests = hr_dataset.plan_site_downloads(_site(), fetch=fake_fetch)
    by_product = {r["product"]: r for r in requests}
    assert "2022" in by_product["rgb"]["url"]
    assert by_product["rgb"]["dest"].startswith("hr_downloads/rgb/")
    assert by_product["dsm"]["year"] == "2021"
    assert by_product["dsm"]["sheet"] == CELL
    assert sum(1 for r in requests if r["product"] == "cop30") == 1


# --------------------------------------------------------------------------- #
# build_tile products
# --------------------------------------------------------------------------- #

def test_build_tile_products_and_ndsm_handling(tmp_path, monkeypatch):
    root = tmp_path / "data"
    files = _synthetic_downloads(root)
    monkeypatch.setattr(hr_dataset, "CHECKSUM_INDEX",
                        _checksum_index(tmp_path, files))
    monkeypatch.setattr(hr_dataset, "REPO_ROOT", tmp_path)
    result = hr_dataset.build_tile(
        _tile(), downloads_root=root / "hr_downloads", out_root=root / "hr",
        water_geometries=[_water_box()], gsd_m=1.0)

    assert result["dsm_year"] == "2021"
    assert result["rgb_year"] == "2019"
    assert result["temporal_flag"] is False  # 2021 - 2019 = 2 <= 4
    assert float(result["datum_offset_m"]) == pytest.approx(2.7, abs=0.05)
    assert "negative_ndsm" in result["flags"]

    out = root / "hr" / "urban" / "testsite_11"
    ndsm, transform, shape, crs = _read_tif(out.with_name(
        "testsite_11_ndsm.tif"))
    assert ndsm.min() >= -hr_dataset.DEFAULT_NDSM_NEG_TOL_M
    assert np.all(ndsm[14:24, 20:30] == 0.0)  # clipped -2.0 patch is now 0

    mask, _, _, _ = _read_tif(out.with_name("testsite_11_valid_mask.tif"))
    assert np.all(mask[13:21, 9:17] == 0)  # water box
    assert mask[0, 0] == 1
    assert 0.0 < float(result["valid_frac"]) < 1.0

    # every product shares the tile grid
    names = ("rgb", "dsm_hr", "dtm_hr", "ndsm", "dem", "prior_90m",
             "prior_250m", "residual", "valid_mask")
    for name in names:
        _, t, s, c = _read_tif(out.with_name(f"testsite_11_{name}.tif"))
        assert t == transform and s == shape and c == crs

    # prior_90m is the benchmark degradation path applied to the coarse dem
    dem, _, _, _ = _read_tif(out.with_name("testsite_11_dem.tif"))
    prior, _, _, _ = _read_tif(out.with_name("testsite_11_prior_90m.tif"))
    expected = evaluation._degrade_reference(dem.astype("float64"), 90.0, 1.0)
    np.testing.assert_allclose(prior, expected.astype("float32"),
                               rtol=1e-4, atol=1e-3, equal_nan=True)

    dsm, _, _, _ = _read_tif(out.with_name("testsite_11_dsm_hr.tif"))
    dtm, _, _, _ = _read_tif(out.with_name("testsite_11_dtm_hr.tif"))
    residual, _, _, _ = _read_tif(out.with_name("testsite_11_residual.tif"))
    np.testing.assert_allclose(residual, (dsm - prior).astype("float32"),
                               rtol=1e-4, atol=1e-3, equal_nan=True)
    assert np.isfinite(dsm).all() and np.isfinite(dtm).all()


def test_build_tile_mask_excludes_nodata(tmp_path, monkeypatch):
    root = tmp_path / "data"
    files = _synthetic_downloads(root, rgb_width=24)
    monkeypatch.setattr(hr_dataset, "CHECKSUM_INDEX",
                        _checksum_index(tmp_path, files))
    monkeypatch.setattr(hr_dataset, "REPO_ROOT", tmp_path)
    result = hr_dataset.build_tile(
        _tile(), downloads_root=root / "hr_downloads", out_root=root / "hr",
        gsd_m=1.0)
    mask, _, _, _ = _read_tif(root / "hr" / "urban"
                              / "testsite_11_valid_mask.tif")
    assert mask[:, :10].all()  # rgb covers the west side
    assert mask[:, 12:].sum() == 0  # no rgb -> nodata -> invalid
    assert float(result["valid_frac"]) < 1.0


def test_manifest_row_validates_with_harness(tmp_path, monkeypatch):
    root = tmp_path / "data"
    files = _synthetic_downloads(root)
    monkeypatch.setattr(hr_dataset, "CHECKSUM_INDEX",
                        _checksum_index(tmp_path, files))
    monkeypatch.setattr(hr_dataset, "REPO_ROOT", tmp_path)
    result = hr_dataset.build_tile(
        _tile(), downloads_root=root / "hr_downloads", out_root=root / "hr",
        gsd_m=1.0)
    row = hr_dataset.manifest_row(result, _site())
    path = tmp_path / "manifest_hr.csv"
    hr_dataset.write_manifest(path, [row])

    required = set(evaluation._MANIFEST_REQUIRED_COLUMNS)
    assert required <= set(hr_dataset.MANIFEST_COLUMNS)

    rows, validation = evaluation.load_manifest(path)
    assert validation["errors"] == []
    assert rows[0]["site_id"] == "testsite"
    assert rows[0]["reference_hr_path"] == result["dsm_path"]
    assert rows[0]["stratum"] == "urban"


def test_shipped_hr_manifest_validates():
    path = hr_dataset.MANIFEST_HR
    if not path.exists():
        pytest.skip("hr dataset not built yet")
    rows, validation = evaluation.load_manifest(path, guard_km=10.0)
    assert validation["errors"] == []
    assert len(rows) == 240
    site_splits = {}
    for row in rows:
        site_splits.setdefault(row["site_id"], set()).add(row["split"])
        assert (evaluation._REPO_ROOT / row["reference_hr_path"]).exists()
        assert row["datum_offset_m"]
    assert all(len(splits) == 1 for splits in site_splits.values())
    # 6 Phase-2 tiles (napf/samedan) + 28 extension tiles on sheets whose
    # newest swissSURFACE3D epoch predates the newest SWISSIMAGE scene.
    assert sum(1 for r in rows if r["temporal_flag"] == "True") == 34
