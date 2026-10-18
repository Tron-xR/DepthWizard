"""Smoke test for the opt-in RS height backends (CPU-friendly).

Architecture + end-to-end verification only: loads one real tile (read-only,
absolute path) or a synthetic urban scene, runs depth.infer_ndsm_metres
through the exact production code path (rs_height windowed inference), and
reports shape/stats/timing. With --synthetic-urban and an altimap backend the
result is range-checked against the model card (heights are nDSM metres >= 0;
GAMUS's tallest real objects are ~151 m LiDAR / 66-104 m predicted, so a sane
output sits in [0, 200) m with some building signal). NOT a benchmark - see
docs/RS_HEIGHT_EVAL_DESIGN.md for the gated ndsm/dsm evaluation modes.

Run one backend per process (the depth model cache is global). The compute
device is set explicitly via DEPTHWIZARD_RS_DEVICE (default cpu here: never
compete with the other session's GPU work); the production default stays
auto (cuda when torch.cuda.device_count() > 0).

    .venv-rs\\Scripts\\python.exe experiments\\rs_eval\\smoke_rs_height.py \\
        --backend altimap_v1 --synthetic-urban --gsd-in 1.0

Caches (HF_HOME/TORCH_HOME) default to the worktree-local dirs so the smoke
never touches shared caches; weights are downloaded on first run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "server"))

BACKEND_ENV = {
    "rs3dada": "DEPTHWIZARD_RS3DADA_MODEL",
    "altimap_v1": "DEPTHWIZARD_ALTIMAP_V1_MODEL",
    "altimap_v2": "DEPTHWIZARD_ALTIMAP_V2_MODEL",
}

# Model card (Dilavesh/altimap-height, GAMUS test): nDSM metres >= 0; tallest
# real-world objects predicted 66-104 m (LiDAR 151 m). Anything >= 200 m or a
# flat zero field is a broken backend, not a hard scene.
_CARD_RANGE_M = (0.0, 200.0)
_CARD_MIN_SIGNAL_M = 1.0


def synthetic_urban(size: int = 171, seed: int = 5) -> "object":
    """Buildings (bright roofs) + SE shadows + roads on textured ground.

    171 px at 1 m/px upsamples to exactly one 518 px window at the altimap
    0.33 m/px target - enough structure to check the range without a GPU.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 0.0, dtype="float32")
    img[:] = np.array([108.0, 112.0, 104.0])          # dry ground base
    img += rng.normal(0.0, 5.0, img.shape)
    img[size // 2 - 4:size // 2 + 4, :, :] = [128.0, 128.0, 130.0]   # road
    img[:, size // 3 - 3:size // 3 + 3, :] = [126.0, 126.0, 128.0]   # road
    buildings = [(0.08, 0.08, 0.30, 0.22), (0.55, 0.10, 0.82, 0.30),
                 (0.15, 0.55, 0.40, 0.80), (0.60, 0.55, 0.90, 0.85),
                 (0.42, 0.35, 0.55, 0.48)]
    for i, (y0, x0, y1, x1) in enumerate(buildings):
        ya, yb = int(y0 * size), int(y1 * size)
        xa, xb = int(x0 * size), int(x1 * size)
        img[ya + 4:yb + 4, xa + 4:xb + 4] *= 0.55     # cast shadow (SE)
        roof = 185.0 + 10.0 * i                        # per-building brightness
        img[ya:yb, xa:xb] = [roof, roof - 4.0, roof - 8.0]

    try:
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(img, sigma=(0.8, 0.8, 0.0))
    except ImportError:
        pass                                           # sharp is fine too
    return np.clip(img, 0, 255).astype("uint8")


def load_rgb(path: Path, crop: int) -> "object":
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        band_count = min(src.count, 3)
        arr = src.read(list(range(1, band_count + 1))).astype("float32")
        if band_count == 1:
            arr = np.repeat(arr, 3, axis=0)
        if arr.max() > 255:              # 16-bit imagery: scale to uint8
            arr = arr * (255.0 / 65535.0)
        rgb = np.transpose(arr, (1, 2, 0)).astype("uint8")
    h, w = rgb.shape[:2]
    if crop and crop < min(h, w):
        y = (h - crop) // 2
        x = (w - crop) // 2
        rgb = rgb[y:y + crop, x:x + crop]
    return np.ascontiguousarray(rgb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=sorted(BACKEND_ENV))
    ap.add_argument("--tile", help="RGB GeoTIFF (read-only); omit for synthetic")
    ap.add_argument("--synthetic-urban", action="store_true",
                    help="run on a generated tile with buildings instead")
    ap.add_argument("--synthetic-size", type=int, default=171,
                    help="synthetic tile edge in px (171 -> one 518 window "
                         "after the 0.33 m/px upsample)")
    ap.add_argument("--crop", type=int, default=518,
                    help="centre-crop a real tile to N x N px (0 = full tile)")
    ap.add_argument("--gsd-in", type=float, default=None,
                    help="input GSD m/px; altimap resamples to its 0.33 target")
    ap.add_argument("--device", default="cpu",
                    help="DEPTHWIZARD_RS_DEVICE for this process (default cpu)")
    ap.add_argument("--window", type=int, default=518)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "rs_eval"
                                         / "smoke_out"))
    args = ap.parse_args()
    if not args.tile and not args.synthetic_urban:
        ap.error("pass --tile or --synthetic-urban")

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf-cache"))
    os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".torch-cache"))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["DEPTHWIZARD_RS_DEVICE"] = args.device

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)

    from app import config  # noqa: E402
    from app.pipeline import depth

    for slug, attr, repo, _fn, _note in depth.RS_BACKENDS:
        setattr(config, attr, repo if slug == args.backend else "")

    if args.synthetic_urban:
        rgb = synthetic_urban(args.synthetic_size)
        source = f"synthetic-urban({args.synthetic_size}px)"
    else:
        rgb = load_rgb(Path(args.tile), args.crop)
        source = args.tile
    print(f"{source} -> rgb {rgb.shape} {rgb.dtype}, backend {args.backend}, "
          f"gsd_in {args.gsd_in}, window {args.window}, device {args.device}",
          flush=True)

    t_load = time.time()
    model = depth.ensure_model()
    load_s = time.time() - t_load
    slug = depth.backend_slug()
    assert slug == args.backend, f"dispatch mismatch: {slug} != {args.backend}"
    assert depth.active_output_kind() == "ndsm_metres"

    from app.pipeline import rs_height  # noqa: E402

    t0 = time.time()
    pred = depth.infer_ndsm_metres(rgb, gsd_in=args.gsd_in)
    elapsed = time.time() - t0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "synthetic_urban" if args.synthetic_urban else Path(args.tile).stem
    npy = out_dir / f"{args.backend}_{stem}_ndsm.npy"
    np.save(npy, pred)
    if args.synthetic_urban:
        np.save(out_dir / "synthetic_urban_rgb.npy", rgb)

    # model grid the backend actually ran at (before the back-resample)
    scale = 1.0
    model_gsd = getattr(model, "model_gsd", None)
    if args.gsd_in and model_gsd:
        scale = args.gsd_in / model_gsd
    model_h = round(rgb.shape[0] * scale)
    n_windows = (len(rs_height._window_starts(model_h, args.window,
                                              args.window // 2)) ** 2
                 if rgb.shape[0] == rgb.shape[1] else None)

    result = {
        "backend": slug,
        "identifier": depth.model_identifier(),
        "license_note": getattr(model, "license_note", None),
        "source": str(source),
        "crop": list(rgb.shape[:2]),
        "gsd_in": args.gsd_in,
        "model_gsd": model_gsd,
        "model_grid": model_h,
        "dtype": str(pred.dtype),
        "min_m": float(np.nanmin(pred)),
        "max_m": float(np.nanmax(pred)),
        "mean_m": float(np.nanmean(pred)),
        "nan_count": int(np.count_nonzero(~np.isfinite(pred))),
        "load_seconds": round(load_s, 1),
        "infer_seconds": round(elapsed, 1),
        "windows": n_windows,
        "compute": rs_height.compute_info(),
        "saved": str(npy),
    }
    print(json.dumps(result, indent=2))

    rc = 0
    if args.synthetic_urban and args.backend.startswith("altimap"):
        lo, hi = _CARD_RANGE_M
        ok_range = (lo <= result["min_m"] and result["max_m"] < hi)
        ok_signal = result["max_m"] > _CARD_MIN_SIGNAL_M
        print(f"model-card range check: min/max within [{lo}, {hi}) m: "
              f"{ok_range}; building signal > {_CARD_MIN_SIGNAL_M} m: {ok_signal}")
        if not (ok_range and ok_signal):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
