"""Elevation-space coarse fields (meters) for the RGB-vs-coarse experiment.

For each tile: down-sample the full-res truth DEM by block-mean (factor f_<arm>
from coarse_meta.json), then bicubic-upsample back to tile resolution. These are
the no-model baselines (arm C) and the residual reference for arms B/D:
    residual(pred) = corr(pred_elev - coarse_up, truth - coarse_up).
Also emits tiles_bbox.json: WGS84 bbox per tile (from DEM georef) for the 3DEP
fetch and per-tile ground pitch. Pure cache builder; no modelling.
"""
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from rasterio.warp import transform_bounds

EXP = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode\exp")
OUT = EXP / "coarse_elev"
OUT.mkdir(parents=True, exist_ok=True)


def dem_path(tile_id):
    if "/" in tile_id:
        fam, name = tile_id.split("/", 1)
        return (r"C:\Users\harsh\Downloads\files\DepthWizard\training_data"
                f"\\{fam}\\{name}_dem.tif")
    if tile_id in ("colorado_original", "colorado_north", "colorado_west"):
        return (r"C:\Users\harsh\Downloads\files\geo_test_data"
                f"\\{tile_id}_dem.tif")
    return (r"C:\Users\harsh\Downloads\files\terrain_stratum_test\data"
            f"\\{tile_id}_dsm.tif")


def pitch_m(tr, crs):
    a = abs(tr.a)
    if crs and crs.is_geographic:
        mid = math.radians((tr.f + tr.c) / 2.0)
        return a * 111320.0 * math.cos(mid)
    return a


def main():
    meta = json.loads((EXP / "coarse_meta.json").read_text())
    # merge the 10 non-trainset tiles (heldout dsm / geo_test dem paths)
    tiles = {m["tile"]: m for m in meta}
    bbox = {}
    for m in meta:
        with rasterio.open(dem_path(m["tile"])) as s:
            arr = s.read(1).astype("float64")
            nodata = s.nodata
            tr, crs = s.transform, s.crs
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            pm = pitch_m(tr, crs)
        bbox4326 = list(transform_bounds(crs, "EPSG:4326", *s.bounds))
        if crs and not crs.is_geographic:
            pm = pitch_m(tr, crs)
        bbox[m["tile"]] = dict(wgs84=bbox4326, epsg4326_pitch=pm)
        h, w = arr.shape
        for res in ("90", "250"):
            f = m[f"f_B{res}"]
            blk = ndimage.uniform_filter(np.nan_to_num(arr, nan=np.nan),
                                         size=f, mode="nearest")[::f, ::f]
            blk = ndimage.zoom(blk, (h / blk.shape[0], w / blk.shape[1]),
                               order=3, mode="nearest")
            npy = OUT / f"{m['tile'].replace('/', '_')}_{res}.npy"
            np.save(npy, blk.astype("float32"))
        print(f"{m['tile']:<28} bbox={[round(v,4) for v in bbox4326]} "
              f"pitch={pm:6.2f}m f90={m['f_B90']} f250={m['f_B250']} "
              f"shape={h}x{w}", flush=True)

    (EXP / "tiles_bbox.json").write_text(json.dumps(bbox, indent=2))
    print(f"saved {len(bbox)} tile bboxes + coarse_elev fields")


if __name__ == "__main__":
    main()