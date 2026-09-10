"""Offline training of a small decoder head on top of FROZEN Depth Anything V2
(small). Produces decoder_training/checkpoints/da2_decoder_finetuned.pt.

Data: training_data/{hilly,forested,sparse}/{name}_{rgb,dem}.tif (copied from
the gather stage; never overlaps the held-out evaluation sites).

Usage (from repo root, using the server venv):
    python decoder_training/train_decoder.py --epochs 5          # smoke test
    python decoder_training/train_decoder.py --epochs 30         # full run
    python decoder_training/train_decoder.py --epochs 30 --resume  # continue

Design:
- Backbone frozen end-to-end (requires_grad=False on every parameter).
- Decoder: 4 conv layers (64->32->16->1) on the backbone's raw depth map.
- Loss: L1 + 0.5 * Sobel-gradient-magnitude L1 (edge sharpness only, no GAN).
- 85/15 train/val split BY REGION (never by pixel) to prevent leakage.
- Target: per-tile min-max normalized DEM in 0..1 (relative depth, matching
  this project's rDSM convention).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "server"
sys.path.insert(0, str(SERVER))

from app.pipeline.decoder_head import (  # noqa: E402
    DecoderHead,
    MODEL_ID,
    init_weights,
    to_pixel_values,
)

CROP = 256
TRAIN_CROPS_PER_REGION = 4
VAL_CROPS_PER_REGION = 4
CKPT_DIR = REPO / "decoder_training" / "checkpoints"
MANIFEST = REPO / "training_data" / "manifest.csv"

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=torch.float32,
).reshape(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(-2, -1)


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    gx = F.conv2d(x, _SOBEL_X.to(x.device), padding=1)
    gy = F.conv2d(x, _SOBEL_Y.to(x.device), padding=1)
    return (gx * gx + gy * gy + 1e-6).sqrt()


def load_regions() -> list[dict]:
    """Scan training_data/{stratum}/*_{rgb,dem}.tif directly (no manifest
    dependency; the gather stage's csv rows may still be buffered)."""
    regions = []
    root = REPO / "training_data"
    for stratum_dir in sorted(root.iterdir()):
        if not stratum_dir.is_dir():
            continue
        rgb_files = sorted(stratum_dir.glob("*_rgb.tif"))
        for rgb in rgb_files:
            dem = rgb.with_name(rgb.name.replace("_rgb.tif", "_dem.tif"))
            if dem.exists():
                regions.append(dict(name=rgb.stem[:-4], stratum=stratum_dir.name,
                                    rgb=str(rgb), dem=str(dem)))
    return regions


def read_dem(path: str) -> tuple[np.ndarray, float]:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr).astype("float64")
    return arr, nodata


def region_statistics(regions: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop regions that are invalid, nodata-heavy, or flat; return usable ones
    plus a list of (name, reason) skips."""
    usable, skips = [], []
    for r in regions:
        dem, _ = read_dem(r["dem"])
        finite = np.isfinite(dem)
        frac = float(finite.mean())
        if frac < 0.7:
            skips.append((r["name"], f"nodata/invalid {frac:.0%} valid"))
            continue
        vals = dem[finite]
        if vals.max() - vals.min() < 1e-3:
            skips.append((r["name"], "flat DEM"))
            continue
        usable.append(r)
    return usable, skips


class PairDataset:
    """Iterates random (train) or fixed-grid (val) crops shared between the RGB
    and DEM of one region. Target is the per-tile min-max normalized DEM."""

    def __init__(self, regions: list[dict], split: str, seed: int):
        self.regions = sorted(regions, key=lambda r: r["name"])
        self.split = split
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.regions) * (TRAIN_CROPS_PER_REGION if self.split == "train"
                                    else VAL_CROPS_PER_REGION)

    def _crop(self, arr, x, y):
        return arr[y:y + CROP, x:x + CROP]

    def _target(self, dem, x, y):
        crop = self._crop(dem, x, y).astype("float64").copy()
        finite = np.isfinite(crop)
        if not finite.any():
            crop[:] = 0.0
        else:
            lo, hi = crop[finite].min(), crop[finite].max()
            if hi - lo < 1e-6:
                crop[:] = 0.0
            else:
                crop = (crop - lo) / (hi - lo)
                crop[~finite] = 0.0
        return torch.from_numpy(crop[None]).float()

    def __getitem__(self, idx):
        ri = idx // (TRAIN_CROPS_PER_REGION if self.split == "train"
                     else VAL_CROPS_PER_REGION)
        ci = idx % (TRAIN_CROPS_PER_REGION if self.split == "train"
                    else VAL_CROPS_PER_REGION)
        r = self.regions[ri]
        rgb = np.asarray(Image.open(r["rgb"]).convert("RGB")).astype("uint8")
        dem, _ = read_dem(r["dem"])
        h, w = dem.shape[:2]
        if self.split == "train":
            x = self.rng.randrange(0, w - CROP + 1)
            y = self.rng.randrange(0, h - CROP + 1)
        else:  # fixed 2x2 grid of center crops
            ncx, ncy = (w - CROP) // 2, (h - CROP) // 2
            x = ci % 2 * ncx if ncx > 0 else 0
            y = ci // 2 * ncy if ncy > 0 else 0
        target = self._target(dem, x, y)
        rgb_crop = np.ascontiguousarray(self._crop(rgb, x, y))
        return to_pixel_values(rgb_crop, self.proc), target

    proc = None


def set_processor(proc) -> None:
    PairDataset.proc = proc


def run_epoch(model, head, ds, device, batch, optimizer=None, desc=""):
    model.eval()
    head.train(mode=optimizer is not None)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch,
                                         shuffle=optimizer is not None,
                                         num_workers=0)
    total, n = 0.0, 0
    for bi, (pv, tgt) in enumerate(loader):
        pv = pv.to(device)
        tgt = tgt.to(device)
        with torch.no_grad():
            depth = model(pixel_values=pv).predicted_depth
        pred = head(depth.clamp(0, 50))
        if tgt.shape[-2:] != pred.shape[-2:]:
            dy = (tgt.shape[-2] - pred.shape[-2]) // 2
            dx = (tgt.shape[-1] - pred.shape[-1]) // 2
            tgt = tgt[:, :, dy:dy + pred.shape[-2], dx:dx + pred.shape[-1]]
        l1 = F.l1_loss(pred, tgt)
        sm = sobel_mag(pred)
        st = sobel_mag(tgt)
        loss = l1 + 0.5 * F.l1_loss(sm, st)
        if not torch.isfinite(loss):
            continue
        total += float(loss.item()) * len(pv)
        n += len(pv)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, head, ds, device, batch):
    return run_epoch(model, head, ds, device, batch, optimizer=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    seed_all = args.seed
    random.seed(seed_all)
    np.random.seed(seed_all % (2**32))
    torch.manual_seed(seed_all)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if args.batch > 8:
        print("cap batch at 8 for the 4GB GPU")
        args.batch = 8
    batch = args.batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} batch={batch} lr={args.lr} crops={CROP}")

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    set_processor(proc)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    n_frozen = sum(p.numel() for p in model.parameters())
    n_backbone_train = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"backbone frozen params={n_frozen:,} trainable={n_backbone_train:,}")

    head = DecoderHead()
    head.apply(init_weights)
    head = head.to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"decoder head params={head_params:,}")

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        head.load_state_dict(ck["state_dict"])
        start_epoch = int(ck["epoch"]) + 1
        print(f"resumed from {args.resume} (epoch {start_epoch})")

    regions = load_regions()
    usable, skips = region_statistics(regions)
    names = {r["name"] for r in usable}
    for name, reason in skips:
        print(f"skip {name}: {reason}")

    names = sorted(names)
    random.Random(seed_all).shuffle(names)
    n_val = max(1, int(round(len(names) * 0.15)))
    val_names, train_names = set(names[:n_val]), set(names[n_val:])
    print(f"regions: total={len(names)} train={len(train_names)} "
          f"val={len(val_names)}")
    print("val regions:", sorted(val_names))

    train_ds = PairDataset([r for r in usable if r["name"] in train_names],
                           "train", seed_all)
    val_ds = PairDataset([r for r in usable if r["name"] in val_names],
                         "val", seed_all)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    history = []
    best_ckpt = (float("inf"), None)  # (best_val_loss, epoch)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, head, train_ds, device, batch,
                               optimizer=optimizer, desc="train")
        val_loss = evaluate(model, head, val_ds, device, batch)
        dt = time.time() - t0
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} ({dt:.0f}s)")
        history.append((epoch, train_loss, val_loss))

        ckpt = dict(state_dict=head.state_dict(), epoch=epoch,
                    train_loss=train_loss, val_loss=val_loss,
                    val_regions=sorted(val_names))
        path = CKPT_DIR / "da2_decoder_finetuned.pt"
        if val_loss < best_ckpt[0]:
            best_ckpt = (val_loss, epoch)
            tmp = CKPT_DIR / "da2_decoder_finetuned.pt.tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, path)
            print(f"  saved best {path} (epoch {epoch}, val {val_loss:.4f})")
        else:
            print(f"  kept best (epoch {best_ckpt[1]}, val {best_ckpt[0]:.4f})")

    with open(REPO / "decoder_training" / "loss_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        w.writerows(history)
    print(f"final: last={history[-1] if history else None} "
          f"best_val={min((v for _, _, v in history), default=None):.4f}")


if __name__ == "__main__":
    main()