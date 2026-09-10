"""Re-run the terrain-stratum comparison against the 4th backend
(fine-tuned DA2 decoder) on the exact same held-out test sites used for
pix2pix / Depth Anything V2 / IMELE in docs/14-evaluation-results.md.

Scoring is the same independent sklearn/scipy method as terrain_stratum_test/
run_backend.py (least-squares affine calibration + Pearson on calibrated preds).

Prints a leakage cross-check: confirms none of the 5 held-out bounding boxes
overlap any region that ended up in training_data.

Usage (server venv, from repo root):
    python decoder_training/eval_finetuned.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO = Path(__file__).resolve().parents[1]
SERVER = str(REPO / "server")
sys.path.insert(0, SERVER)

import app.config as config  # noqa: E402

CKPT = os.environ.get("DEPTHWIZARD_FINETUNED_MODEL", "") or str(
    REPO / "decoder_training" / "checkpoints" / "da2_decoder_finetuned.pt")
config.FINETUNED_MODEL_PATH = CKPT

from app.pipeline.depth import infer_relative_dsm  # noqa: E402

DATA = str(Path(__file__).resolve().parents[1].parent / "terrain_stratum_test" / "data")
G = str(REPO / "sample_images" / "gamus_test")

# Held-out eval sites (west, south, east, north): must NOT be in training_data.
HELD_OUT = [
    ("hilly_colorado", (-105.6, 39.9, -105.4, 40.1)),
    ("hilly_uttarakhand", (78.0, 30.35, 78.2, 30.55)),
    ("hilly_alps", (7.8, 46.6, 8.0, 46.8)),
    ("gamus_dc", None),  # fixed sample tile; never in training_data by construction
]


def overlaps(a, b):
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return (min(ae, be) - max(aw, bw) > 1e-9) and (min(an, bn) - max(as_, bs) > 1e-9)


def leakage_check():
    """Scan training_data from disk; get each region's true bbox from the
    GeoTIFF bounds, and confirm none overlap a held-out eval site."""
    root = REPO / "training_data"
    if not root.exists():
        print("WARNING: no training_data (training set not gathered yet)")
        return
    regions = []
    for stratum in sorted(p for p in root.iterdir() if p.is_dir()):
        for rgb in sorted(stratum.glob("*_rgb.tif")):
            dem = rgb.with_name(rgb.name.replace("_rgb.tif", "_dem.tif"))
            if not dem.exists():
                continue
            with rasterio.open(dem) as src:
                b = src.bounds
            regions.append((rgb.name, (b.left, b.bottom, b.right, b.top)))
    print(f"leakage check: {len(regions)} training regions vs held-out sites...")
    for tag, box in HELD_OUT:
        if box is None:
            print(f"  {tag}: excluded by construction (local sample tile)")
            continue
        hit = [name for name, b in regions if overlaps(b, box)]
        status = "OK (no overlap)" if not hit else f"LEAK: {hit}"
        print(f"  {tag}: {status}")


def cmpt(rgb, dem, tag):
    with rasterio.open(dem) as s:
        t = s.read(1).astype(float)
    p = infer_relative_dsm(np.array(Image.open(rgb).convert("RGB"))).astype(float)
    if p.shape != t.shape:
        p = np.array(Image.fromarray(p).resize((t.shape[1], t.shape[0]),
                                               Image.BILINEAR))
    v = np.isfinite(p) & np.isfinite(t)
    pv, tv = p[v], t[v]
    A = np.vstack([pv, np.ones_like(pv)]).T
    a, b = np.linalg.lstsq(A, tv, rcond=None)[0]
    pc = a * pv + b
    rmse = float(np.sqrt(mean_squared_error(tv, pc)))
    std = float(tv.std())
    return (round(float(pearsonr(pv, tv)[0]), 4),
            round(float(pearsonr(pc, tv)[0]), 4),
            round(rmse, 2),
            round(float(mean_absolute_error(tv, pc)), 2),
            round(rmse / std, 3))


def main():
    print(f"backend: finetuned DA2 decoder  (ckpt={CKPT})")
    leakage_check()
    print()
    print(f"{'site':<18}{'raw_corr':>10}{'calib_corr':>12}{'RMSE':>10}{'MAE':>9}{'NRMSE':>8}")
    for name in ["hilly_colorado", "hilly_uttarakhand", "hilly_alps"]:
        raw, cal, rmse, mae, nrmse = cmpt(f"{DATA}\\{name}_rgb.tif",
                                          f"{DATA}\\{name}_dsm.tif", name)
        print(f"{name:<18}{raw:>10}{cal:>12}{rmse:>10}{mae:>9}{nrmse:>8}")
    gor = os.path.join
    if os.path.exists(gor(G, "DC_03_26_RGB.png")) and os.path.exists(gor(G, "DC_03_26_AGL.tif")):
        raw, cal, rmse, mae, nrmse = cmpt(gor(G, "DC_03_26_RGB.png"),
                                          gor(G, "DC_03_26_AGL.tif"), "gamus")
        print(f"{'gamus':<18}{raw:>10}{cal:>12}{rmse:>10}{mae:>9}{nrmse:>8}")


if __name__ == "__main__":
    main()