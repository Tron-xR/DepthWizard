"""High-resolution Swiss dataset build (Phase 2).

Pipeline: ``sites`` -> ``download`` -> ``build``:

1. Site design: 28 independent sites, 7 per stratum (urban / sparse alpine
   bare / hilly / forest), split train/val/test BY SITE from a fixed seed.
   The committed site table is ``server/data/hr_sites.csv``; test sites are
   frozen (see tests/test_hr_dataset.py).
2. Download (resumable, sha256-indexed via hr_download): SWISSIMAGE DOP10
   (RGB), swissSURFACE3D Raster (DSM), swissALTI3D (DTM) from the swisstopo
   STAC, Copernicus GLO-30 from the public AWS bucket (reusing
   dem_source._tile_prefix), and swissTLMregio (water polygons).
3. Build: per-tile float32 GeoTIFFs on one working grid (EPSG:2056, 1 m
   default, 512x512 default): rgb, dsm_hr, dtm_hr, ndsm, coarse dem
   ({tile_id}_dem.tif), prior_90m, prior_250m (degraded coarse dem, same code
   path as the Phase 1 benchmark baselines), residual, valid mask.
   Per-tile acquisition years, datum offset (recorded, never silently
   removed) and flags land in ``server/data/manifest_hr.csv``.

Deliberate simplifications (ponytail):
- per product sheet the NEWEST STAC epoch is chosen (swisstopo republishes
  whole 1 km sheets per epoch); per-tile years record what was actually used;
- water = TLMregio generalized polygons (lakes + wide rivers); narrow rivers
  are not buffered - upgrade path: TLM3D water lines with a buffer;
- the datum offset is estimated on flat, vegetation/building-free pixels
  (|ndsm| < 2 m, slope < 15%); tiles without such pixels fall back to the
  site median and record the method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.pipeline import dem_source, hr_download
from app.pipeline.evaluation import _degrade_reference

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "server" / "data"
SITES_CSV = DATA_DIR / "hr_sites.csv"
MANIFEST_HR = DATA_DIR / "manifest_hr.csv"
CHECKSUM_INDEX = DATA_DIR / "hr_downloads.csv"
DOWNLOADS_ROOT = DATA_DIR / "hr_downloads"
HR_ROOT = DATA_DIR / "hr"

WORK_CRS = "EPSG:2056"
STRATA = ("urban", "sparse", "hilly", "forest")
SPLIT_SIZES = {"test": 2, "val": 2, "train": 3}
SEED = 42
DEFAULT_GSD_M = 1.0
DEFAULT_TILE_PX = 512
TILES_PER_AXIS = 2
DEFAULT_MIN_SITE_DISTANCE_KM = 2.0
DEFAULT_TEMPORAL_GAP_YEARS = 4
DEFAULT_NDSM_NEG_TOL_M = 0.5
DATUM_FLAT_MAX_NDSM_M = 2.0
DATUM_FLAT_MAX_SLOPE = 0.15
DATUM_MIN_FLAT_PX = 512

STAC_ENDPOINT = "https://data.geo.admin.ch/api/stac/v0.9"
PRODUCTS = {
    "rgb": {"collection": "ch.swisstopo.swissimage-dop10",
            "assets": ("_0.1_2056.tif", "_2_2056.tif"),
            "res_m": 0.1},
    "dsm": {"collection": "ch.swisstopo.swisssurface3d-raster",
            "assets": ("_0.5_2056_5728.tif",),
            "res_m": 0.5},
    "dtm": {"collection": "ch.swisstopo.swissalti3d",
            "assets": ("_0.5_2056_5728.tif", "_2_2056_5728.tif"),
            "res_m": 0.5},
}
WATER_COLLECTION = "ch.swisstopo.swisstlmregio"
WATER_ASSETS = ("_2056.gpkg.zip", "_2056.shp.zip", "_2056.gdb.zip")
WATER_LAYER_HINTS = ("wasser", "gewaesser", "see", "fluss", "bach")

# Site design: well-known Swiss locations, stratum assigned from the dominant
# land cover / terrain of the ~1 km footprint (rationale column records why).
# Frozen after commit: test sites are pinned in the test suite.
SITES = (
    ("zurich_center", "urban", 8.5417, 47.3769, "dense city core, mixed roofs"),
    ("basel_center", "urban", 7.5886, 47.5596, "city core at the Rhine"),
    ("bern_center", "urban", 7.4474, 46.9480, "old town, dense blocks"),
    ("geneva_center", "urban", 6.1432, 46.2044, "city core left of the lake"),
    ("lausanne_center", "urban", 6.6323, 46.5197, "steep city above the lake"),
    ("winterthur", "urban", 8.7241, 47.4985, "city core, industrial edges"),
    ("luzern_center", "urban", 8.3093, 47.0502, "city core at the lake outlet"),
    ("zermatt", "sparse", 7.7486, 46.0207, "alpine village, bare valley walls"),
    ("saas_fee", "sparse", 7.9300, 46.1100, "high valley, bare slopes"),
    ("samedan", "sparse", 9.8712, 46.5340, "open Engadin valley floor"),
    ("grimsel", "sparse", 8.3350, 46.5620, "bare granite high plateau"),
    ("simplon", "sparse", 8.0300, 46.2500, "bare alpine pass slopes"),
    ("arolla", "sparse", 7.4800, 46.0300, "high glacial valley, moraine"),
    ("susten", "sparse", 8.4400, 46.7300, "bare pass terrain, scree"),
    ("emmental", "hilly", 7.7000, 46.9500, "rolling farmland hills"),
    ("appenzell", "hilly", 9.4000, 47.3300, "steep meadows, scattered farms"),
    ("maggia_valley", "hilly", 8.7000, 46.2500, "steep Ticino valley walls"),
    ("chasseral", "hilly", 7.0500, 47.1300, "Jura ridge, pastures"),
    ("la_dole", "hilly", 6.1000, 46.4200, "Jura crest, open slopes"),
    ("napf", "hilly", 7.9400, 47.0100, "pre-alpine hill country"),
    ("gruyeres", "hilly", 7.0800, 46.5800, "pre-alpine pastures"),
    ("sihlwald", "forest", 8.4800, 47.2700, "dense beech forest"),
    ("bachtel", "forest", 8.8700, 47.3000, "forested ridge, mixed stands"),
    ("vaud_jura", "forest", 6.3500, 46.6000, "continuous Jura forest"),
    ("gantrisch", "forest", 7.4500, 46.7200, "forested pre-alpine slopes"),
    ("val_colla", "forest", 9.1200, 46.0800, "southern broadleaf forest"),
    ("aegeri", "forest", 8.6000, 47.1300, "forest above the lake"),
    ("randen", "forest", 8.5500, 47.7500, "wooded Randen plateau"),
)


# --------------------------------------------------------------------------- #
# Sites + split assignment
# --------------------------------------------------------------------------- #

def wgs84_to_lv95(lon: float, lat: float) -> tuple:
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform("EPSG:4326", WORK_CRS, [lon], [lat])
    return float(xs[0]), float(ys[0])


def lv95_to_wgs84(east: float, north: float) -> tuple:
    from rasterio.warp import transform as warp_transform

    lons, lats = warp_transform(WORK_CRS, "EPSG:4326", [east], [north])
    return float(lons[0]), float(lats[0])


def assign_splits(sites, *, sizes=SPLIT_SIZES, seed: int = SEED) -> dict:
    """Split BY SITE: within each stratum, shuffle deterministically and take
    test/val/train slices. One tile never crosses a split."""
    rng = np.random.default_rng(seed)
    assignment = {}
    for stratum in STRATA:
        ids = sorted(s["site_id"] for s in sites if s["stratum"] == stratum)
        expected = sum(sizes.values())
        if len(ids) != expected:
            raise ValueError(
                f"stratum {stratum}: {len(ids)} sites, expected {expected}")
        order = [ids[i] for i in rng.permutation(len(ids))]
        start = 0
        for split in ("test", "val", "train"):
            take = sizes[split]
            for site_id in order[start:start + take]:
                assignment[site_id] = split
            start += take
    return assignment


def build_site_rows(*, seed: int = SEED) -> list:
    sites = [{"site_id": sid, "stratum": stratum, "lon": lon, "lat": lat,
              "rationale": rationale}
             for sid, stratum, lon, lat, rationale in SITES]
    assignment = assign_splits(sites, seed=seed)
    rows = []
    for site in sites:
        east, north = wgs84_to_lv95(site["lon"], site["lat"])
        rows.append({"site_id": site["site_id"], "stratum": site["stratum"],
                     "split": assignment[site["site_id"]],
                     "lon_wgs84": site["lon"], "lat_wgs84": site["lat"],
                     "e_lv95": round(east), "n_lv95": round(north),
                     "rationale": site["rationale"]})
    return rows


def site_distance_m(a, b) -> float:
    return math.hypot(a["e_lv95"] - b["e_lv95"], a["n_lv95"] - b["n_lv95"])


def validate_sites(rows, *, min_distance_km: float = DEFAULT_MIN_SITE_DISTANCE_KM):
    """Structural checks for the committed site table."""
    errors = []
    if len(rows) < 25:
        errors.append(f"only {len(rows)} sites, need >= 25")
    seen = set()
    for row in rows:
        if row["site_id"] in seen:
            errors.append(f"duplicate site_id {row['site_id']}")
        seen.add(row["site_id"])
        if row["stratum"] not in STRATA:
            errors.append(f"{row['site_id']}: bad stratum {row['stratum']}")
        if row["split"] not in SPLIT_SIZES:
            errors.append(f"{row['site_id']}: bad split {row['split']}")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = site_distance_m(rows[i], rows[j])
            if d < min_distance_km * 1000:
                errors.append(
                    f"sites {rows[i]['site_id']} and {rows[j]['site_id']} are "
                    f"{d / 1000:.2f} km apart (< {min_distance_km} km)")
    return errors


def write_sites_csv(path, rows, *, min_distance_km=DEFAULT_MIN_SITE_DISTANCE_KM):
    errors = validate_sites(rows, min_distance_km=min_distance_km)
    if errors:
        raise ValueError("site table invalid:\n  " + "\n  ".join(errors))
    fields = ("site_id", "stratum", "split", "lon_wgs84", "lat_wgs84",
              "e_lv95", "n_lv95", "rationale")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["site_id"]))


def load_sites_csv(path=SITES_CSV) -> list:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = [{k: (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh)]
    for row in rows:
        row["lon_wgs84"] = float(row["lon_wgs84"])
        row["lat_wgs84"] = float(row["lat_wgs84"])
        row["e_lv95"] = int(row["e_lv95"])
        row["n_lv95"] = int(row["n_lv95"])
    return rows


# --------------------------------------------------------------------------- #
# Tile layout (working grid: single GSD, EPSG:2056)
# --------------------------------------------------------------------------- #

def site_tiles(site, *, gsd_m: float = DEFAULT_GSD_M,
               tile_px: int = DEFAULT_TILE_PX, per_axis: int = TILES_PER_AXIS):
    """Tiles of one site: per_axis x per_axis grid centred on the site point,
    snapped to the GSD grid. All tiles inherit the site's split."""
    from rasterio.transform import from_origin

    size = tile_px * gsd_m
    footprint = size * per_axis
    x0 = round((site["e_lv95"] - footprint / 2.0) / gsd_m) * gsd_m
    y_top = round((site["n_lv95"] + footprint / 2.0) / gsd_m) * gsd_m
    tiles = []
    for j in range(per_axis):
        for i in range(per_axis):
            tx = x0 + i * size
            ty = y_top - j * size
            tiles.append({
                "tile_id": f"{site['site_id']}_{j}{i}",
                "site_id": site["site_id"],
                "stratum": site["stratum"],
                "split": site["split"],
                "x0": tx,
                "y0": ty,
                "width": tile_px,
                "height": tile_px,
                "gsd_m": gsd_m,
                "transform": from_origin(tx, ty, gsd_m, gsd_m),
                "bounds": (tx, ty - size, tx + size, ty),
            })
    return tiles


def sheet_cells(xmin, ymin, xmax, ymax) -> list:
    """The 1 km product sheets ("E-N", kilometre indices) covering a bbox."""
    cells = []
    for e in range(int(math.floor(xmin / 1000)), int(math.floor((xmax - 1e-9) / 1000)) + 1):
        for n in range(int(math.floor(ymin / 1000)), int(math.floor((ymax - 1e-9) / 1000)) + 1):
            cells.append(f"{e}-{n}")
    return cells


# --------------------------------------------------------------------------- #
# STAC planning
# --------------------------------------------------------------------------- #

def _stac_get(url, *, fetch=None):
    fetch = fetch or urllib.request.urlopen
    req = urllib.request.Request(url, headers={"User-Agent": hr_download.USER_AGENT})
    with fetch(req, timeout=60) as response:
        return json.load(response)


def stac_items(collection: str, bbox_wgs84, *, fetch=None) -> list:
    """All STAC items of a collection intersecting a WGS84 bbox."""
    minx, miny, maxx, maxy = bbox_wgs84
    url = (f"{STAC_ENDPOINT}/collections/{collection}/items"
           f"?bbox={minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}&limit=100")
    features = []
    while url:
        data = _stac_get(url, fetch=fetch)
        features.extend(data.get("features", []))
        nxt = [link["href"] for link in data.get("links", [])
               if link.get("rel") == "next"]
        url = nxt[0] if nxt else None
    return features


def item_year(feature) -> str:
    stamp = (feature.get("properties") or {}).get("datetime") or ""
    if stamp:
        return stamp[:4]
    for token in (feature.get("id") or "").split("_"):
        if len(token) == 4 and token.isdigit():
            return token
    return ""


def item_sheet(feature) -> str:
    """Sheet cell from the item id (…_YYYY_E-N)."""
    return item_sheet_name(feature.get("id") or "")


def pick_asset(feature, tokens) -> str:
    for token in tokens:
        for asset in (feature.get("assets") or {}).values():
            href = asset.get("href") or ""
            if token in href:
                return href
    return ""


def choose_items(features, cells, *, newest=True) -> dict:
    """One item per needed sheet: the newest epoch published for that cell."""
    chosen = {}
    for feature in features:
        sheet = item_sheet(feature)
        if sheet not in cells:
            continue
        year = item_year(feature)
        current = chosen.get(sheet)
        if current is None:
            chosen[sheet] = feature
        elif newest and year > item_year(current):
            chosen[sheet] = feature
    return chosen


def plan_site_downloads(site_row, *, fetch=None) -> list:
    """Download requests for one site: rgb/dsm/dtm sheets + GLO-30 cells."""
    requests = []
    tiles = site_tiles(site_row)
    xmin = min(t["bounds"][0] for t in tiles)
    ymin = min(t["bounds"][1] for t in tiles)
    xmax = max(t["bounds"][2] for t in tiles)
    ymax = max(t["bounds"][3] for t in tiles)
    cells = sorted(set(sheet_cells(xmin, ymin, xmax, ymax)))
    minx, miny = lv95_to_wgs84(xmin, ymin)
    maxx, maxy = lv95_to_wgs84(xmax, ymax)
    if minx > maxx:
        minx, maxx = maxx, minx
    bbox_wgs84 = (minx, miny, maxx, maxy)
    for product, spec in PRODUCTS.items():
        chosen = choose_items(stac_items(spec["collection"], bbox_wgs84,
                                         fetch=fetch), cells)
        for sheet, feature in sorted(chosen.items()):
            url = pick_asset(feature, spec["assets"])
            if not url:
                continue
            requests.append({
                "product": product, "sheet": sheet,
                "year": item_year(feature), "url": url,
                "dest": f"{DOWNLOADS_ROOT.name}/{product}/{url.rsplit('/', 1)[-1]}",
                "site_id": site_row["site_id"],
            })
    # Copernicus GLO-30: public AWS bucket, one file per 1-degree cell
    lat_cells = {int(math.floor(lat)) for lat in (miny, maxy - 1e-9)}
    lon_cells = {int(math.floor(lon)) for lon in (minx, maxx - 1e-9)}
    prefixes = sorted({dem_source._tile_prefix(float(lat), float(lon))
                       for lat in lat_cells for lon in lon_cells})
    for prefix in prefixes:
        requests.append({
            "product": "cop30", "sheet": prefix,
            "year": "", "url": f"{dem_source._BASE_COP}/{prefix}/{prefix}.tif",
            "dest": f"{DOWNLOADS_ROOT.name}/cop30/{prefix}.tif",
            "site_id": site_row["site_id"],
        })
    return requests


def plan_water_download(*, fetch=None) -> dict:
    features = stac_items(WATER_COLLECTION, (5.9, 45.8, 10.5, 47.9),
                          fetch=fetch)
    latest = sorted(features, key=lambda f: f["id"])[-1]
    url = pick_asset(latest, WATER_ASSETS)
    return {"product": "water", "sheet": latest["id"], "year": item_year(latest),
            "url": url,
            "dest": f"{DOWNLOADS_ROOT.name}/water/{url.rsplit('/', 1)[-1]}",
            "site_id": "-"}


# --------------------------------------------------------------------------- #
# Download execution
# --------------------------------------------------------------------------- #

def dest_abs(dest_rel: str) -> Path:
    return REPO_ROOT / "server" / dest_rel


def run_downloads(requests, *, index_path=CHECKSUM_INDEX, workers: int = 4,
                  urlopen=None, log=print):
    """Fetch every request, updating the sha256 index. Requests are deduped by
    destination (sites share 1 km sheets)."""
    index = hr_download.load_checksums(index_path)
    unique = {}
    for request in requests:
        unique.setdefault(request["dest"], request)
    records = []
    failures = []

    def one(request):
        dest = dest_abs(request["dest"])
        expected = index.get(request["dest"], {}).get("sha256") or None
        record = hr_download.download_file(
            request["url"], dest, expected_sha256=expected, urlopen=urlopen,
            log=log)
        record.update({"product": request["product"], "sheet": request["sheet"],
                       "year": request["year"], "url": request["url"]})
        record["path"] = request["dest"]
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, r): r for r in unique.values()}
        for future, request in futures.items():
            try:
                record = future.result()
                records.append(record)
                reused = "cached" if record["reused"] else "fetched"
                log(f"[{reused}] {request['dest']} "
                    f"({record['size_bytes'] / 1e6:.1f} MB)")
            except Exception as exc:  # noqa: BLE001 - collected per file
                failures.append((request["dest"], str(exc)))
                log(f"[failed] {request['dest']}: {exc}")
    merged = dict(index)
    for record in records:
        merged[record["path"]] = record
    hr_download.save_checksums(index_path, merged.values())
    if failures:
        raise RuntimeError(
            f"{len(failures)} downloads failed:\n  "
            + "\n  ".join(f"{d}: {e}" for d, e in failures))
    return records


# --------------------------------------------------------------------------- #
# Building tiles
# --------------------------------------------------------------------------- #

def _product_files(downloads_root, product: str) -> dict:
    """{sheet: path} for one product's downloaded files."""
    files = {}
    root = Path(downloads_root) / product
    for path in sorted(root.glob("*")):
        if path.name.endswith(".part"):
            continue
        sheet = item_sheet_name(path.name)
        files.setdefault(sheet, path)
    return files


def item_sheet_name(name: str) -> str:
    for token in name.split("_"):
        parts = token.split("-")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return token
    return ""


def _reproject_into(dst, dst_transform, src_path, *, src_bbox=None, bands=(1,),
                    src_crs=None, dst_crs=WORK_CRS, resampling="bilinear"):
    """Read a window of src_path and reproject it into dst (nan fill)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds

    mode = {"bilinear": Resampling.bilinear, "cubic": Resampling.cubic,
            "nearest": Resampling.nearest}[resampling]
    with rasterio.open(src_path) as src:
        if src_bbox is None:
            height, width = dst.shape[-2:]
            bbox = (dst_transform.c, dst_transform.f - height * dst_transform.a,
                    dst_transform.c + width * dst_transform.a, dst_transform.f)
            bbox = rasterio.warp.transform_bounds(dst_crs, src.crs, *bbox)
        else:
            bbox = rasterio.warp.transform_bounds(src_crs or dst_crs, src.crs, *src_bbox)
        window = from_bounds(*bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(indexes=list(bands), window=window, boundless=True,
                        masked=True)
        src_transform = src.window_transform(window)
        for band_index in range(len(bands)):
            source = data[band_index].filled(np.nan).astype("float64")
            destination = dst if dst.ndim == 2 else dst[band_index]
            reproject(source=source, destination=destination,
                      src_transform=src_transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=mode, init_dest_nodata=False)
    return dst


def tile_bounds_wgs84(transform, width, height):
    import rasterio.warp

    x0, y0 = transform.c, transform.f
    x1, y1 = x0 + width * transform.a, y0 + height * transform.e
    bounds = rasterio.warp.transform_bounds(WORK_CRS, "EPSG:4326", x0, y1, x1, y0)
    return bounds  # minx, miny, maxx, maxy


def estimate_datum_offset(dsm, dtm, ndsm, coarse):
    """Median (dsm - coarse) on flat, vegetation/building-free pixels."""
    gy, gx = np.gradient(dtm)
    slope = np.hypot(gx, gy)
    flat = (np.isfinite(ndsm) & np.isfinite(coarse) & np.isfinite(dsm)
            & (np.abs(ndsm) < DATUM_FLAT_MAX_NDSM_M)
            & (slope < DATUM_FLAT_MAX_SLOPE))
    if int(flat.sum()) < DATUM_MIN_FLAT_PX:
        return None, "insufficient_flat_pixels"
    offset = float(np.median(dsm[flat] - coarse[flat]))
    return offset, "tile_flat"


def load_water_polygons(water_root, *, bbox=None):
    """Water polygon geometries from the TLMregio product (GPKG/shp)."""
    import fiona

    root = Path(water_root)
    candidates = [p for p in sorted(root.rglob("*")) if p.suffix in (".gpkg",)]
    if not candidates:
        unpacked = unpack_water_archive(root)
        candidates = [unpacked] if unpacked else []
    if not candidates:
        raise FileNotFoundError(f"no TLMregio vector file under {root}")
    path = candidates[-1]
    geometries = []
    for layer in fiona.listlayers(path):
        if not any(hint in layer.lower() for hint in WATER_LAYER_HINTS):
            continue
        with fiona.open(path, layer=layer, bbox=bbox) as collection:
            for feature in collection:
                geometry = feature.get("geometry")
                if geometry and geometry.get("type") in (
                        "Polygon", "MultiPolygon", "GeometryCollection"):
                    geometries.append(geometry)
    return geometries


def unpack_water_archive(root):
    """Unzip the TLMregio archive next to the download on first use."""
    import zipfile

    root = Path(root)
    existing = sorted(root.rglob("*.gpkg"))
    if existing:
        return existing[0]
    archives = sorted(root.rglob("*.zip"))
    if not archives:
        return None
    archive = archives[-1]
    target = archive.parent / archive.name[:-4].rsplit(".", 1)[0]
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    gpkgs = sorted(target.rglob("*.gpkg"))
    return gpkgs[0] if gpkgs else None


def rasterize_water(geometries, transform, shape):
    from rasterio.features import rasterize

    if not geometries:
        return np.zeros(shape, dtype=bool)
    burned = rasterize([(g, 1) for g in geometries], out_shape=shape,
                       transform=transform, fill=0, dtype="uint8")
    return burned.astype(bool)


def build_tile(tile, *, downloads_root=DOWNLOADS_ROOT, out_root=HR_ROOT,
               water_geometries=None, gsd_m=DEFAULT_GSD_M,
               ndsm_neg_tol_m=DEFAULT_NDSM_NEG_TOL_M,
               temporal_gap_years=DEFAULT_TEMPORAL_GAP_YEARS) -> dict:
    """Build every raster product of one tile on the working grid."""
    import rasterio

    downloads_root = Path(downloads_root)
    out_dir = Path(out_root) / tile["stratum"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (tile["height"], tile["width"])
    transform = tile["transform"]
    x0, y0, x1, y1 = tile["bounds"]

    rgb = np.full((3,) + shape, np.nan, dtype="float32")
    dsm = np.full(shape, np.nan, dtype="float32")
    dtm = np.full(shape, np.nan, dtype="float32")
    coarse = np.full(shape, np.nan, dtype="float32")
    years = {"rgb": set(), "dsm": set(), "dtm": set()}

    sheet_files = {p: _product_files(downloads_root, p)
                   for p in ("rgb", "dsm", "dtm")}
    cells = sheet_cells(x0, y0, x1, y1)
    checksum_index = hr_download.load_checksums(CHECKSUM_INDEX)

    def _note_years(product, files):
        for path in files:
            record = checksum_index.get(f"{DOWNLOADS_ROOT.name}/{product}/{path.name}")
            if record is not None and record.get("year"):
                years[product].add(record["year"])

    for product, dst, bands in (("rgb", rgb, (1, 2, 3)), ("dsm", dsm, (1,)),
                                ("dtm", dtm, (1,))):
        files = [sheet_files[product][sheet] for sheet in cells
                 if sheet in sheet_files[product]]
        for path in files:
            _reproject_into(dst, transform, path, bands=bands)
        _note_years(product, files)

    cop_files = sorted((downloads_root / "cop30").glob("*.tif"))
    if cop_files:
        wgs84 = tile_bounds_wgs84(transform, tile["width"], tile["height"])
        for path in cop_files:
            _reproject_into(coarse, transform, path, src_bbox=wgs84,
                            src_crs="EPSG:4326", resampling="bilinear")

    water = rasterize_water(water_geometries or [], transform, shape)

    ndsm = dsm - dtm
    ndsm_negative = np.isfinite(ndsm) & (ndsm < -ndsm_neg_tol_m)
    ndsm = np.where(ndsm_negative, 0.0, ndsm).astype("float32")
    ndsm_negative_frac = (float(ndsm_negative.sum())
                          / max(int(np.isfinite(dsm).sum()), 1))

    prior_90 = _degrade_reference(coarse, 90.0, gsd_m)
    prior_250 = _degrade_reference(coarse, 250.0, gsd_m)
    residual = (dsm - prior_90).astype("float32")

    offset, offset_method = estimate_datum_offset(dsm, dtm, ndsm, coarse)

    finite = (np.isfinite(dsm) & np.isfinite(dtm) & np.isfinite(coarse)
              & np.isfinite(rgb).all(axis=0) & np.isfinite(prior_90)
              & ~water)
    valid_mask = finite.astype("uint8")

    gap = 0
    if all(years[p] for p in ("rgb", "dsm", "dtm")):
        all_years = [int(y) for p in years for y in years[p]]
        gap = max(all_years) - min(all_years)
    temporal_flag = gap > temporal_gap_years

    flags = []
    if temporal_flag:
        flags.append("temporal_mismatch")
    if ndsm_negative_frac > 0.005:
        flags.append("negative_ndsm")

    rgb_year = max(years["rgb"]) if years["rgb"] else ""
    dsm_year = max(years["dsm"]) if years["dsm"] else ""
    dtm_year = max(years["dtm"]) if years["dtm"] else ""

    paths = {}
    def _write(name, array, count, dtype="float32", nodata=np.nan):
        path = out_dir / f"{tile['tile_id']}_{name}.tif"
        profile = dict(driver="GTiff", height=shape[0], width=shape[1],
                       count=count, dtype=dtype, crs=WORK_CRS, transform=transform,
                       nodata=nodata, compress="deflate")
        if dtype == "float32":
            profile["predictor"] = 3
        with rasterio.open(path, "w", **profile) as dst:
            if count == 1:
                dst.write(array, 1)
            else:
                dst.write(array, indexes=list(range(1, count + 1)))
        paths[name] = path
        return path

    _write("rgb", rgb, 3)
    _write("dsm_hr", dsm, 1)
    _write("dtm_hr", dtm, 1)
    _write("ndsm", ndsm, 1)
    _write("dem", coarse, 1)  # coarse prior on the working grid (harness name)
    _write("prior_90m", prior_90.astype("float32"), 1)
    _write("prior_250m", prior_250.astype("float32"), 1)
    _write("residual", residual, 1)
    _write("valid_mask", valid_mask, 1, dtype="uint8", nodata=None)

    bounds = tile_bounds_wgs84(transform, tile["width"], tile["height"])
    valid_frac = float(valid_mask.sum()) / valid_mask.size
    return {
        "tile_id": tile["tile_id"], "site_id": tile["site_id"],
        "stratum": tile["stratum"], "split": tile["split"],
        "x0": int(tile["x0"]), "y0": int(tile["y0"]),
        "width": tile["width"], "height": tile["height"],
        "rgb_path": _rel(paths["rgb"]), "dsm_path": _rel(paths["dsm_hr"]),
        "dtm_path": _rel(paths["dtm_hr"]), "ndsm_path": _rel(paths["ndsm"]),
        "coarse_dem_path": _rel(paths["dem"]),
        "prior_90m_path": _rel(paths["prior_90m"]),
        "prior_250m_path": _rel(paths["prior_250m"]),
        "residual_path": _rel(paths["residual"]),
        "mask_path": _rel(paths["valid_mask"]),
        "rgb_year": rgb_year, "dsm_year": dsm_year, "dtm_year": dtm_year,
        "temporal_gap_years": gap, "temporal_flag": temporal_flag,
        "datum_offset_m": (f"{offset:.3f}" if offset is not None else ""),
        "datum_offset_method": offset_method,
        "ndsm_negative_frac": f"{ndsm_negative_frac:.6f}",
        "water_frac": f"{float(water.sum()) / water.size:.6f}",
        "valid_frac": f"{valid_frac:.6f}",
        "flags": ";".join(flags),
        "bounds_wgs84": ",".join(f"{v:.6f}" for v in bounds),
    }


def _rel(path) -> str:
    return Path(path).resolve().relative_to(Path(REPO_ROOT).resolve()).as_posix()


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

MANIFEST_COLUMNS = (
    "tile_id", "site_id", "region", "split", "sensor", "gsd", "crs",
    "reference_product", "reference_type", "reference_res_m",
    "vertical_datum", "stratum", "capture_date", "reference_hr_path", "notes",
    "bounds_wgs84", "contaminated_arms",
    "rgb_path", "dsm_path", "dtm_path", "ndsm_path", "coarse_dem_path",
    "prior_90m_path", "prior_250m_path", "residual_path", "mask_path",
    "rgb_year", "dsm_year", "dtm_year", "temporal_gap_years", "temporal_flag",
    "datum_offset_m", "datum_offset_method", "ndsm_negative_frac", "water_frac",
    "valid_frac", "flags", "x0", "y0", "width", "height",
)


def manifest_row(tile_result, site_row, *, res_m: float = 0.5) -> dict:
    return {
        "tile_id": tile_result["tile_id"],
        "site_id": tile_result["site_id"],
        "region": tile_result["site_id"],
        "split": tile_result["split"],
        "sensor": "SWISSIMAGE DOP10",
        "gsd": f"{tile_result.get('gsd_m', DEFAULT_GSD_M):g}",
        "crs": WORK_CRS,
        "reference_product": "swissSURFACE3D Raster",
        "reference_type": "DSM",
        "reference_res_m": f"{res_m:g}",
        "vertical_datum": "LN02",
        "stratum": tile_result["stratum"],
        "capture_date": "",
        "reference_hr_path": tile_result["dsm_path"],
        "notes": f"{site_row.get('rationale', '')}; heights LN02 vs EGM2008 "
                 "coarse (see datum_offset_m)",
        "bounds_wgs84": tile_result["bounds_wgs84"],
        "contaminated_arms": "",
        **{k: tile_result[k] for k in (
            "rgb_path", "dsm_path", "dtm_path", "ndsm_path", "coarse_dem_path",
            "prior_90m_path", "prior_250m_path", "residual_path", "mask_path",
            "rgb_year", "dsm_year", "dtm_year", "temporal_gap_years",
            "temporal_flag", "datum_offset_m", "datum_offset_method",
            "ndsm_negative_frac", "water_frac", "valid_frac", "flags",
            "x0", "y0", "width", "height")},
    }


def write_manifest(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS,
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["tile_id"]))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cmd_sites(args) -> int:
    rows = build_site_rows(seed=args.seed)
    write_sites_csv(args.out, rows, min_distance_km=args.min_site_distance_km)
    counts = {}
    for row in rows:
        counts[(row["stratum"], row["split"])] = counts.get((row["stratum"], row["split"]), 0) + 1
    print(f"[sites] {len(rows)} sites -> {args.out}")
    for stratum in STRATA:
        line = ", ".join(f"{split}:{counts.get((stratum, split), 0)}"
                         for split in ("train", "val", "test"))
        print(f"  {stratum:8s} {line}")
    test_ids = sorted(r["site_id"] for r in rows if r["split"] == "test")
    print("[sites] test sites (frozen):", ", ".join(test_ids))
    return 0


def cmd_download(args) -> int:
    rows = load_sites_csv(args.sites_csv)
    requests = []
    for row in rows:
        requests.extend(plan_site_downloads(row))
    requests.append(plan_water_download())
    records = run_downloads(requests, index_path=args.index,
                            workers=args.workers)
    total = sum(r["size_bytes"] for r in records) / 1e6
    print(f"[download] {len(records)} files, {total:.0f} MB this run")
    return 0


def cmd_build(args) -> int:
    rows = load_sites_csv(args.sites_csv)
    water_geometries = None
    try:
        water_geometries = load_water_polygons(args.downloads_root / "water",
                                               bbox=None)
        print(f"[build] water polygons: {len(water_geometries)}")
    except Exception as exc:  # noqa: BLE001 - water stays all-false
        print(f"[build] water unavailable ({exc}); mask ignores water")
    manifest_rows = []
    site_by_id = {row["site_id"]: row for row in rows}
    for row in rows:
        for tile in site_tiles(row, gsd_m=args.gsd, tile_px=args.tile_px):
            result = build_tile(tile, downloads_root=args.downloads_root,
                                out_root=args.out_root,
                                water_geometries=water_geometries,
                                gsd_m=args.gsd,
                                ndsm_neg_tol_m=args.ndsm_neg_tol,
                                temporal_gap_years=args.temporal_gap_years)
            result["gsd_m"] = args.gsd
            manifest_rows.append(manifest_row(result, site_by_id[row["site_id"]]))
            print(f"[build] {tile['tile_id']} valid={result['valid_frac']} "
                  f"offset={result['datum_offset_m'] or '-'} "
                  f"flags={result['flags'] or '-'}")
    write_manifest(args.manifest, manifest_rows)
    temporal = sum(1 for r in manifest_rows if r["temporal_flag"])
    print(f"[build] {len(manifest_rows)} tiles -> {args.manifest} "
          f"(temporal-flagged: {temporal})")
    return 0


def cmd_checksums(args) -> int:
    records = hr_download.collect_checksums(args.downloads_root)
    server_root = REPO_ROOT / "server"
    index = hr_download.load_checksums(args.index)
    for record in records:
        record["path"] = (Path(record["path"]).resolve()
                          .relative_to(server_root).as_posix())
        old = index.get(record["path"]) or {}
        record["url"] = old.get("url", "")
        record["product"] = old.get("product", "")
        record["sheet"] = old.get("sheet", "")
        record["year"] = old.get("year", "")
    hr_download.save_checksums(args.index, records)
    problems = hr_download.verify_checksums(args.index, args.downloads_root)
    print(f"[checksums] {len(records)} files indexed, {len(problems)} problems")
    for problem in problems:
        print(" ", problem)
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sites = sub.add_parser("sites", help="write the committed site table")
    p_sites.add_argument("--out", type=Path, default=SITES_CSV)
    p_sites.add_argument("--min-site-distance-km", type=float,
                         default=DEFAULT_MIN_SITE_DISTANCE_KM)
    p_sites.set_defaults(func=cmd_sites)

    p_dl = sub.add_parser("download", help="fetch all products")
    p_dl.add_argument("--sites-csv", type=Path, default=SITES_CSV)
    p_dl.add_argument("--index", type=Path, default=CHECKSUM_INDEX)
    p_dl.add_argument("--workers", type=int, default=4)
    p_dl.set_defaults(func=cmd_download)

    p_build = sub.add_parser("build", help="build tiles + manifest_hr.csv")
    p_build.add_argument("--sites-csv", type=Path, default=SITES_CSV)
    p_build.add_argument("--downloads-root", type=Path, default=DOWNLOADS_ROOT)
    p_build.add_argument("--out-root", type=Path, default=HR_ROOT)
    p_build.add_argument("--manifest", type=Path, default=MANIFEST_HR)
    p_build.add_argument("--gsd", type=float, default=DEFAULT_GSD_M)
    p_build.add_argument("--tile-px", type=int, default=DEFAULT_TILE_PX)
    p_build.add_argument("--ndsm-neg-tol", type=float,
                         default=DEFAULT_NDSM_NEG_TOL_M)
    p_build.add_argument("--temporal-gap-years", type=int,
                         default=DEFAULT_TEMPORAL_GAP_YEARS)
    p_build.set_defaults(func=cmd_build)

    p_sum = sub.add_parser("checksums", help="regenerate + verify the index")
    p_sum.add_argument("--downloads-root", type=Path, default=DOWNLOADS_ROOT)
    p_sum.add_argument("--index", type=Path, default=CHECKSUM_INDEX)
    p_sum.set_defaults(func=cmd_checksums)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
