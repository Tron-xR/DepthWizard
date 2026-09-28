"""Per-tile elevation-space metrics for arms B/D (checkpoints) and C.

Model arms (B90/B250/D90/D250, per seed): OOF tile -> its fold's best model;
extras -> fold 0's model. Pred is relative depth 0..1 -> affine calibration
per tile (fit on the 80% split).
C arms (C90/C250): pred == bicubic-upsampled coarse (meters).

--truth native  (default): truth = the tile's own COP30/Tilezen DEM.
--truth dem3dep: truth = USGS 3DEP export aligned to the pred grid
                 (independent reference, CONUS tiles only).
Writes runs/metrics_[3dep_]arm[_{seed}].csv
    arm,seed,tile,fam,src,block,mae,rmse,raw_r,resid_r,raw_pos,n_valid
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import rasterio

EXP = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode\exp")
sys.path.insert(0, str(EXP))

import metrics_lib  # noqa: E402
import exp_train  # noqa: E402
from exp_model import build, checkpoints_out, predict  # noqa: E402


def rgb_path(tile_id):
    if "/" in tile_id:
        fam, name = tile_id.split("/", 1)
        return metrics_lib.REPO / "training_data" / fam / f"{name}_rgb.tif"
    if tile_id in ("colorado_original", "colorado_north", "colorado_west"):
        return metrics_lib.GEOTEST / f"{tile_id}_rgb.tif"
    return metrics_lib.HELDOUT / f"{tile_id}_rgb.tif"


def dep3(tile_id):
    p = EXP / "dem3dep" / f"{tile_id.replace('/', '_')}.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as s:
        arr = s.read(1).astype("float64")
        tr, crs = s.transform, s.crs
    arr = np.where((arr == 0.0) | (arr < -1000.0), np.nan, arr)
    return arr, tr, crs


def align_to_pred(dep, pred, tile_id):
    from app.pipeline import evaluation, uploader
    _, meta = uploader.load_raster(rgb_path(tile_id))
    arr, tr, crs = dep
    a = evaluation.align_dem_to_prediction(
        pred, arr,
        predicted_transform=meta.get("transform"), predicted_crs=meta.get("crs"),
        dem_transform=tr, dem_crs=crs)
    if a.shape != pred.shape:
        from PIL import Image
        a = np.asarray(Image.fromarray(a).resize(
            (pred.shape[1], pred.shape[0]), Image.BILINEAR))
    return a


class Truth:
    def __init__(self, mode):
        self.mode = mode

    def score(self, pred, tile_id, affine):
        if self.mode == "native":
            t = metrics_lib.load_truth(tile_id)
        else:
            dep = dep3(tile_id)
            if dep is None:
                return None
            t = align_to_pred(dep, pred, tile_id)
        cu = metrics_lib.coarse_up(tile_id, self.res)
        return metrics_lib.score(pred, t, cu, tile_id, affine=affine)

    def set_res(self, res):
        self.res = res


def block_of(tile_id, folds):
    name = tile_id.split("/")[1] if "/" in tile_id else tile_id
    c = folds["cells"].get(name)
    return (c["ci"] // folds["side"], c["cj"] // folds["side"]) if c else ""


def model_rows(arm, seed, truth):
    folds = json.loads((EXP / "folds.json").read_text())
    all_tiles = {tid: exp_train.tile_new(tid, fam, src, rgbp, demp, arm)
                 for tid, fam, src, rgbp, demp in exp_train.tiles()}
    device = torch_device()
    truth.set_res(arm[1:])
    val_of = {}
    for f in range(folds["K"]):
        for cid in folds["folds"][str(f)]["val"]:
            val_of[f"{cid.split('_')[0]}/{cid}"] = f
    extras = [tid for tid, t in all_tiles.items() if t["src"] != "trainset"]
    rows = []
    for f in range(folds["K"]):
        model, head, _, proc = build(arm, device)
        ck = torch_load(checkpoints_out(arm, seed, f))
        head.load_state_dict(ck["head"])
        if ck["stem"]:
            model.load_state_dict(ck["stem"], strict=False)
        model.eval(); head.eval()
        here = [tid for tid, fof in val_of.items() if fof == f] + \
            ([tid for tid in extras] if f == 0 else [])
        for tid in sorted(here):
            t = all_tiles[tid]
            pred = predict(model, head, proc, device, t["rgb"], t["coarse"],
                           zero_rgb=arm.startswith("D"))
            m = truth.score(pred, tid, affine=True)
            if m is None:
                continue
            rows.append(dict(arm=arm, seed=seed, tile=tid, fam=t["fam"],
                             src=t["src"], block=block_of(tid, folds), **m))
            print(f"[{arm} s{seed}] {tid:<28} mae={m['mae']:7.1f}m "
                  f"rmse={m['rmse']:7.1f}m raw_r={m['raw_r']:+.3f} "
                  f"resid_r={m['resid_r']:+.3f}", flush=True)
    return rows


def c_rows(arm, truth):
    truth.set_res(arm[1:])
    rows = []
    tiles = sorted(m["tile"] for m in
                   json.loads((EXP / "coarse_meta.json").read_text()))
    for tid in tiles:
        cu = metrics_lib.coarse_up(tid, arm[1:])
        m = truth.score(cu, tid, affine=False)
        if m is None:
            continue
        fam = tid.split("/")[0] if "/" in tid else tid.split("_")[0]
        if "/" in tid:
            src = "trainset"
        elif tid in ("colorado_original", "colorado_north", "colorado_west"):
            src = "geo_test"
        else:
            src = "heldout"
        rows.append(dict(arm=arm, seed="", tile=tid, fam=fam, src=src,
                         block="", **m))
        print(f"[{arm}] {tid:<28} mae={m['mae']:7.1f}m "
              f"rmse={m['rmse']:7.1f}m raw_r={m['raw_r']:+.3f} "
              f"resid_r={m['resid_r']:+.3f}", flush=True)
    return rows


def torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth", choices=("native", "dem3dep"), default="native")
    args = ap.parse_args()
    truth = Truth(args.truth)
    rows = c_rows(args.arm, truth) if args.arm.startswith("C") else \
        model_rows(args.arm, args.seed, truth)
    pre = "metrics3dep_" if args.truth == "dem3dep" else "metrics_"
    name = (f"{pre}{args.arm}.csv" if args.arm.startswith("C")
            else f"{pre}{args.arm}_{args.seed}.csv")
    path = EXP / "runs" / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "seed", "tile", "fam", "src",
                                          "block", "mae", "rmse", "raw_r",
                                          "resid_r", "raw_pos", "n_valid"])
        w.writeheader(); w.writerows(rows)
    print(f"-> {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()