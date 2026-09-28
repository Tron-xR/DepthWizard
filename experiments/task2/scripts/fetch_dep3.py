"""Fetch USGS 3DEP (~10m, keyless) for all CONUS tiles as an independent truth.

Source: USGS nationalmap 3DEPElevation ImageServer exportImage (open, f=image,
pixelType=F32, meters, EPSG:4326). bbox = WGS84 extent of each tile (from
tiles_bbox.json); export size ~ native tile resolution.
Writes dem3dep/{tile}.tif and records which tiles fetched. No GPU.
"""
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio

EXP = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode\exp")
OUT = EXP / "dem3dep"
OUT.mkdir(parents=True, exist_ok=True)
META = json.loads((EXP / "coarse_meta.json").read_text())
BBOX = json.loads((EXP / "tiles_bbox.json").read_text())

URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/"
       "ImageServer/exportImage?bbox={0}&bboxSR=4326&size={1},{2}&imageSR=4326"
       "&format=tiff&pixelType=F32&noData=None&f=image")


def is_conus(w, s, e, n):
    return -130.0 < w < e < -60.0 and 24.0 < s < n < 52.0


def deg_per_px(pitch_m, lat):
    return pitch_m / (111320.0 * max(math.cos(math.radians(lat)), 0.1))


def main():
    targets = {}
    for m in META:
        tid = m["tile"]
        w, s, e, n = BBOX[tid]["wgs84"]
        if not is_conus(w, s, e, n):
            continue
        pitch = BBOX[tid]["epsg4326_pitch"]
        # export at tile-native pitch (all tiles >=~19 m, coarser than 3DEP's
        # ~10 m, so no downsampling of the reference happens)
        dpp = deg_per_px(pitch, (s + n) / 2)
        ncols = max(16, int(math.ceil((e - w) / dpp)))
        nrows = max(16, int(math.ceil((n - s) / dpp)))
        targets[tid] = (w, s, e, n, ncols, nrows)

    done, failed = [], []
    for tid in sorted(targets):
        out = OUT / f"{tid.replace('/', '_')}.tif"
        if out.exists():
            done.append(tid)
            continue
        w, s, e, n, nc, nr = targets[tid]
        url = URL.format(urllib.parse.quote(f"{w:9f},{s:9f},{e:9f},{n:9f}"), nc, nr)
        ok = False
        for attempt in range(8):
            backoff = min(4.0 * (2 ** attempt) + 5, 90.0)
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (research)"})
                r = urllib.request.urlopen(req, timeout=180)
                data = r.read()
                if not (data[:2] in (b"II", b"MM") or data[:1] in (b"\x89", )):
                    raise RuntimeError(
                        f"not a raster; ctype={r.headers.get('Content-Type')} "
                        f"{data[:160]!r}")
                with rasterio.open(BytesIO(data)) as src:
                    a = src.read(1)
                assert a.size > 0 and np.isfinite(a).mean() > 0.5, a.size
                with rasterio.open(out, "w", driver="GTiff", height=a.shape[0],
                                   width=a.shape[1], count=1, dtype="float32",
                                   crs="EPSG:4326",
                                   transform=src.transform) as dst:
                    dst.write(a.astype("float32")[None])
                done.append(tid)
                print(f"OK  {tid:<28} {a.shape[1]}x{a.shape[0]} "
                      f"min={np.nanmin(a):.0f} max={np.nanmax(a):.0f}", flush=True)
                ok = True
                break
            except Exception as ex:  # noqa: BLE001 - transient service errors
                print(f"    retry {tid} [{attempt+1}/8] {type(ex).__name__}"
                      f" {ex!r} (wait {backoff:.0f}s)", flush=True)
                time.sleep(backoff)
        if not ok:
            failed.append(tid)
            print(f"FAIL {tid}", flush=True)
        time.sleep(3.0)

    (EXP / "dem3dep_fetched.json").write_text(json.dumps(
        dict(done=done, failed=failed), indent=2))
    print(f"\nfetched {len(done)}/{len(targets)} CONUS tiles; "
          f"failed={failed}")


if __name__ == "__main__":
    main()