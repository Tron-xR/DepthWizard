"""Download georeferenced COP30 DEM tiles from OpenTopography for the
Colorado test regions. Data-gathering only; not part of the shipped server.

Usage:
    python download_dem.py
Requires: OPENTOPO_API_KEY env var; 'requests'. Outputs {region}_dem.tif.
"""
import os
import sys

import requests

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geo_test_data")
OUT_DIR = os.path.abspath(OUT_DIR)

BASE = "https://portal.opentopography.org/API/globaldem"

REGIONS = {
    "colorado_original": dict(south=39.9, north=40.1, west=-105.6, east=-105.4),
    "colorado_north":   dict(south=40.1, north=40.3, west=-105.6, east=-105.4),
    "colorado_west":    dict(south=39.9, north=40.1, west=-105.8, east=-105.6),
}


def main():
    key = os.environ.get("OPENTOPO_API_KEY", "").strip()
    if not key:
        print("OPENTOPO_API_KEY is not set; refusing to proceed.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, geo in REGIONS.items():
        params = {
            "demtype": "COP30",
            "outputFormat": "GTiff",
            "API_Key": key,
        }
        params.update(geo)
        out_path = os.path.join(OUT_DIR, f"{name}_dem.tif")
        print(f"fetching {name}: {geo}", flush=True)
        with requests.get(BASE, params=params, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
        print(f"  wrote {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


if __name__ == "__main__":
    main()
