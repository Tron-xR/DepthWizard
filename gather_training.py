"""Gather training RGB+DEM pairs (hilly/forested/sparse strata) for the
fine-tuned-decoder experiment. Data-collection only.

Reuses geo_test/download_dem.py and geo_test/download_rgb.py AS-IS (imported,
module attrs overridden at the call site; no core logic rewritten).

Outputs:
    training_data/{stratum}/{name}_dem.tif   COP30 DEM via OpenTopography
    training_data/{stratum}/{name}_rgb.tif   Esri World Imagery warped to DEM grid
    training_data/manifest.csv               one row per region + validation status
    training_data/gather.log                 progress log

Requires: OPENTOPO_API_KEY env var; requests, rasterio, numpy, PIL.
"""
import csv
import io
import math
import os
import sys
import time

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "geo_test"))
import download_dem  # noqa: E402
import download_rgb  # noqa: E402

OUT = os.path.join(ROOT, "training_data")
LOG = os.path.join(OUT, "gather.log")
MANIFEST = os.path.join(OUT, "manifest.csv")

RGB_PIXEL_ZOOM = 13  # 0.2deg box ~ 1152px mosaic > 720px DEM grid; see report

# Evaluation/test sites that must stay HELD OUT (never in training_data).
# (west, south, east, north)
EXCLUDED = [
    ("hilly_colorado_bbox",   (-105.6, 39.9, -105.4, 40.1)),  # == colorado_original
    ("colorado_north",        (-105.6, 40.1, -105.4, 40.3)),
    ("colorado_west",         (-105.8, 39.9, -105.6, 40.1)),
    ("hilly_uttarakhand",     (78.0, 30.35, 78.2, 30.55)),
    ("hilly_alps",            (7.8, 46.6, 8.0, 46.8)),
]

# Candidate boxes per stratum, defined as base-center + offset grids,
# box size 0.2 deg (matches the known-good COP30 tile shape).
def grid(base_lat, base_lon, lat_offs, lon_offs, size=0.2):
    out = []
    for lo in lon_offs:
        for la in lat_offs:
            s = base_lat + la
            w = base_lon + lo
            out.append((round(w, 6), round(s, 6),
                        round(w + size, 6), round(s + size, 6)))  # (w,s,e,n)
    out.sort()
    return out


def overlaps(a, b):
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    x0, x1 = max(aw, bw), min(ae, be)
    y0, y1 = max(as_, bs), min(an, bn)
    return (x1 - x0 > 1e-9) and (y1 - y0 > 1e-9)


def keep(cands, n):
    """Deterministic spread: drop bboxes overlapping HELD OUT sites, then take
    a strided subset of exactly n non-overlapping candidates."""
    kept = [c for c in cands if not any(overlaps(c, ex[1]) for ex in EXCLUDED)]
    if n >= len(kept):
        return kept
    step = len(kept) / n
    picked = [kept[int(i * step)] for i in range(n)]
    for i in range(n - 1):
        assert not overlaps(picked[i], picked[i + 1]), "internal bbox collision"
    return picked


STRATA = {}

STRATA["hilly"] = {
    "base": "colorado_front_range",
    "boxes": keep(grid(40.0, -105.5,
                       lat_offs=[-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7],
                       lon_offs=[-1.6, -1.4, -1.2, -1.0, -0.8, -0.6]),
                  14),
}
STRATA["hilly"].update({
    "base": "colorado_front_range+western_ghats",
    "boxes": STRATA["hilly"]["boxes"]
             + keep(grid(15.5, 74.15,
                         lat_offs=[-0.5, -0.3, -0.1, 0.1, 0.3],
                         lon_offs=[-0.3, -0.1, 0.1, 0.3]),
                    4),
})

STRATA["forested"] = {
    "base": "nh_white_mountains",
    "boxes": keep(grid(44.1, -71.5,
                       lat_offs=[-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
                       lon_offs=[-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]),
                  14),
}
STRATA["forested"].update({
    "base": "nh_white_mountains+appalachians_smokies",
    "boxes": STRATA["forested"]["boxes"]
             + keep(grid(35.7, -83.3,
                         lat_offs=[-0.3, 0.0, 0.3],
                         lon_offs=[-0.4, 0.0, 0.4]),
                    4),
})

STRATA["sparse"] = {
    "base": "n_arizona",
    "boxes": keep(grid(36.1, -110.9,
                       lat_offs=[-0.3, -0.1, 0.1, 0.3],
                       lon_offs=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]),
                  14),
}
STRATA["sparse"].update({
    "base": "n_arizona+mojave+nm",
    "boxes": STRATA["sparse"]["boxes"]
             + keep(grid(36.6, -117.0,
                         lat_offs=[0.1, 0.3],
                         lon_offs=[-0.3, 0.1, 0.5])
             + grid(32.5, -106.1,
                    lat_offs=[0.0],
                    lon_offs=[-0.2, 0.2]),
                    4),
})


def validate(name, sub):
    rgb = os.path.join(sub, f"{name}_rgb.tif")
    dem = os.path.join(sub, f"{name}_dem.tif")
    try:
        with rasterio.open(dem) as s:
            arr = s.read(1)
        with rasterio.open(rgb) as g:
            rarr = g.read()
        ok = arr.size > 0 and rarr.size > 0 and arr.shape == rarr.shape[1:]
        note = "" if ok else "shape mismatch or empty"
        return ok, arr.shape[0], arr.shape[1], os.path.getsize(dem), os.path.getsize(rgb), note
    except Exception as e:  # noqa: BLE001 - collection must not die on one bad file
        return False, 0, 0, 0, 0, repr(e)


# Elevation-tile fallback: AWS "terrain-tiles" (Tilezen terrarium) public
# dataset. RGB-encodes elevation: R*256+G+B/256-32768 = meters. Used ONLY when
# OpenTopography's daily key quota is exhausted (HTTP 401), so DEM collection
# can continue instead of dying at the rate cap. mercator tile math is reused
# verbatim from download_rgb.py.
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def fetch_dem_tilezen(bbox, out_path, z=13):
    w, s, e, n = bbox
    x0, x1, y0, y1 = download_rgb.tile_range((w, s, e, n), z)
    rows = []
    for yt in range(y0, y1 + 1):
        row = []
        for xt in range(x0, x1 + 1):
            r = requests.get(TERRARIUM_URL.format(z=z, y=yt, x=xt), timeout=60)
            r.raise_for_status()
            rgb = np.array(Image.open(io.BytesIO(r.content)).convert("RGB")).astype("float64")
            elev = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
            row.append(elev.astype("float32"))
            time.sleep(0.25)
        rows.append(np.concatenate(row, axis=1))
    mosaic = np.concatenate(rows, axis=0)
    west0, north0, size = download_rgb.tile_bounds_merc(x0, y0, z)
    px = size / 256.0
    with rasterio.open(out_path, "w", driver="GTiff", height=mosaic.shape[0],
                       width=mosaic.shape[1], count=1, dtype="float32",
                       transform=from_origin(west0, north0, px, px),
                       crs="EPSG:3857") as dst:
        dst.write(mosaic[None])
    print(f"  [tilezen fallback] wrote {out_path} ({mosaic.shape[1]}x{mosaic.shape[0]}px)",
          flush=True)


def main():
    key = os.environ.get("OPENTOPO_API_KEY", "").strip()
    if not key:
        print("OPENTOPO_API_KEY not set; refusing to proceed.", file=sys.stderr)
        sys.exit(1)

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    os.makedirs(OUT, exist_ok=True)
    first = not os.path.exists(MANIFEST)
    mf = open(MANIFEST, "a", newline="")
    writer = csv.writer(mf)
    if first:
        writer.writerow(["stratum", "name", "south", "north", "west", "east",
                         "rgb_bytes", "dem_bytes", "valid", "note"])
        mf.flush()
    log = open(LOG, "a")
    log.write(f"\n=== gather start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    counts = {}
    fetched = 0
    for stratum, spec in STRATA.items():
        sub = os.path.join(OUT, stratum)
        os.makedirs(sub, exist_ok=True)
        got = 0
        targets = spec["boxes"]
        log.write(f"[{stratum}] {len(targets)} target boxes (base={spec['base']})\n")
        for i, (w, s, e, n) in enumerate(targets):
            if limit is not None and fetched >= limit:
                break
            name = f"{stratum}_{i:02d}"
            rgb = os.path.join(sub, f"{name}_rgb.tif")
            dem = os.path.join(sub, f"{name}_dem.tif")

            ok, h, wd, db, rb, note = validate(name, sub)
            if ok and os.path.exists(rgb) and os.path.exists(dem):
                last = f"[{name}] already valid ({h}x{wd})"
                log.write(last + "\n")
                print(last, flush=True)
                got += 1
                writer.writerow([stratum, name, s, n, w, e, rb, db, 1, ""])
                mf.flush()
                continue

            print(f"[{name}] fetching DEM bbox=({w},{s},{e},{n})", flush=True)
            fetched += 1
            download_dem.REGIONS = {name: dict(south=s, north=n, west=w, east=e)}
            download_dem.OUT_DIR = sub
            try:
                download_dem.main()
            except Exception as ex:  # noqa: BLE001
                dem_ok = False
                try:
                    fetch_dem_tilezen((w, s, e, n), dem)
                    dem_ok = True
                except Exception as ex2:  # noqa: BLE001
                    log.write(f"[{name}] DEM failed "
                              f"(opentopo {ex!r}; tilezen {ex2!r})\n")
                if not dem_ok:
                    writer.writerow([stratum, name, s, n, w, e, 0, 0, 0,
                                     f"dem:{ex!r}"])
                    mf.flush()
                    time.sleep(2)
                    continue
                log.write(f"[{name}] opentopo failed ({ex!r}); "
                          "used tilezen fallback\n")

            print(f"[{name}] fetching RGB (zoom {RGB_PIXEL_ZOOM})", flush=True)
            download_rgb.OUT_DIR = sub
            download_rgb.ZOOM = RGB_PIXEL_ZOOM
            try:
                download_rgb.fetch_region((w, s, e, n), name=name)
            except Exception as ex:  # noqa: BLE001
                log.write(f"[{name}] RGB fetch failed: {ex!r}\n")
                writer.writerow([stratum, name, s, n, w, e, 0, 0, 0, f"rgb:{ex!r}"])

            ok, h, wd, db, rb, note = validate(name, sub)
            writer.writerow([stratum, name, s, n, w, e, rb, db, int(ok), note])
            mf.flush()
            line = f"[{name}] valid={ok} {h}x{wd} dem={db}B rgb={rb}B {note}"
            log.write(line + "\n")
            print(line, flush=True)
            if ok:
                got += 1
            time.sleep(2)  # 2s between OpenTopography calls to respect rate limits

        counts[stratum] = got
        log.write(f"[{stratum}] VALID PAIRS: {got}\n")

    mf.flush()
    mf.close()
    log.flush()
    log.close()

    print("\n=== SUMMARY ===", flush=True)
    total = 0
    for k, v in STRATA.items():
        print(f"  {k}: {counts[k]}/{len(v['boxes'])} valid", flush=True)
        total += counts[k]
    print(f"  TOTAL valid pairs: {total}", flush=True)
    if total < 50:
        print("  WARNING: under 50 valid pairs; training set is small - see report.",
              flush=True)


if __name__ == "__main__":
    main()