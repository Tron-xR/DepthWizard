#!/usr/bin/env python3
"""Reproducible elevation benchmark for the DepthWizard training tiles.

Protocols (see the benchmark report for the audit rationales):

  historical       Protocol A, legacy per-tile 80/20 PIXEL split: infer on each
                   tile, report raw relative-depth correlation over ALL valid
                   pixels (the /evaluate "relative" mode) and affine-calibrated
                   RMSE/MAE/correlation on the held-out 20% ("calibrated" mode,
                   canonical fit_affine). This mirrors tools/score_my_data.py.
                   When the tiles overlap a backend's training data the numbers
                   are a reproduction/diagnostic, NOT independent validation.
  geographic       Protocol B, geographic holdout: tiles grouped into regions,
                   the affine calibration is fit ONCE on the pooled dev-region
                   pixels and FROZEN for every test region (no test pixels ever
                   enter the fit). Constant dev-median-elevation baseline is
                   scored on the identical test pixels. Test metrics use ALL
                   valid pixels of the test tiles (no fitting => no holdout).
  local_calibration  Protocol B-local: within each region, split tiles into a
                   calibration block and a spatially separated test block (a
                   buffer of >= one tile width between them); fit on the
                   calibration block only, score the test block. This mirrors
                   the deployed per-location calibration workflow (a reference
                   DEM fetched for the same footprint) without pixel leakage.

Run:
  cd server
  uv run python ../tools/benchmark_tiles.py inventory
  uv run python ../tools/benchmark_tiles.py run --protocol historical --backend pix2pix \
      --reference scored_training_full.txt
  uv run python ../tools/benchmark_tiles.py run --protocol geographic --backend pix2pix
  uv run python ../tools/benchmark_tiles.py run --protocol local_calibration --backend pix2pix

Every run writes a NEW directory under benchmark_runs/ (never overwrites) with
manifest.csv, run.json, environment.json, model_info.json, per_tile.csv,
per_region.csv, summary.json, summary.md, and optionally per_tile/<id>.json,
predictions/<id>_relative.npy, comparisons/<id>.png.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_ROOT = REPO_ROOT / "server"
for _p in (str(REPO_ROOT), str(SERVER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_ROOT = REPO_ROOT / "training_data"
DEFAULT_RUN_ROOT = REPO_ROOT / "benchmark_runs"
DEFAULT_FINETUNED_CKPT = REPO_ROOT / "decoder_training" / "checkpoints" / "da2_decoder_finetuned.pt"
DEFAULT_IMELE_MODEL = REPO_ROOT / "models" / "imele_model.tar"
MANIFEST_TEMPLATE = REPO_ROOT / "tools" / "benchmark_manifest.template.csv"

# Verdict thresholds mirror tools/score_my_data.py (raw-vs-calibrated sign flip).
RAW_NEG_THRESHOLD = -0.25
CAL_POS_THRESHOLD = 0.25
RAW_WEAK_THRESHOLD = 0.2
POLARITY_EPS = 0.05

# Region pools: the same candidate-box grids as gather_training.py STRATA
# ("base" name before the "+" is the first pool, after it the next). Full
# candidate grids are used (not the strided keep() pick), so a tile is mapped
# by the pool it actually overlaps - never from its filename.
def _region_pools():
    from gather_training import grid

    return [
        ("colorado_front_range", grid(40.0, -105.5,
                                      [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7],
                                      [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6])),
        ("western_ghats", grid(15.5, 74.15,
                               [-0.5, -0.3, -0.1, 0.1, 0.3],
                               [-0.3, -0.1, 0.1, 0.3])),
        ("nh_white_mountains", grid(44.1, -71.5,
                                    [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
                                    [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])),
        ("appalachians_smokies", grid(35.7, -83.3,
                                      [-0.3, 0.0, 0.3],
                                      [-0.4, 0.0, 0.4])),
        ("n_arizona", grid(36.1, -110.9,
                           [-0.3, -0.1, 0.1, 0.3],
                           [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])),
        ("mojave", grid(36.6, -117.0, [0.1, 0.3], [-0.3, 0.1, 0.5])),
        ("new_mexico", grid(32.5, -106.1, [0.0], [-0.2, 0.2])),
    ]


# Protocol B default split. Dev = two large regions (28 tiles, two continents'
# worth of relief); test = every other region, none adjacent to dev or each
# other. Freeze before scoring; override only via an explicit manifest.
DEFAULT_PROTOCOL_B_SPLIT = {
    "colorado_front_range": "dev",
    "nh_white_mountains": "dev",
    "western_ghats": "test",
    "appalachians_smokies": "test",
    "n_arizona": "test",
    "mojave": "test",
    "new_mexico": "test",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# Files whose content determines the bytes a prediction .npy holds for a given
# input pair + checkpoint. Any edit invalidates every cached prediction.
_INFERENCE_STACK = (
    "tools/benchmark_tiles.py",
    "server/app/pipeline/depth.py",
    "server/app/pipeline/imele.py",
    "server/app/pipeline/decoder_head.py",
    "server/app/pipeline/rs_height.py",
    "server/app/pipeline/rs_dpt.py",
)


def _inference_fingerprint() -> str:
    h = hashlib.sha256()
    for rel in _INFERENCE_STACK:
        h.update(rel.encode())
        h.update((REPO_ROOT / rel).read_bytes())
    return h.hexdigest()


def overlap_area(a, b) -> float:
    x0, x1 = max(a[0], b[0]), min(a[2], b[2])
    y0, y1 = max(a[1], b[1]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_gap_deg(a, b) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return max(dx, dy)


def assign_region(bounds_wgs84) -> Optional[str]:
    best, best_area = None, 0.0
    for name, boxes in _region_pools():
        for box in boxes:
            area = overlap_area(bounds_wgs84, box)
            if area > best_area:
                best, best_area = name, area
    return best


def verdict(raw_c, cal_c) -> dict:
    """Mirror tools/score_my_data.py single-axis flag + two-axis verdicts."""
    flag, note = "ok", None
    if raw_c is not None and cal_c is not None:
        if raw_c < RAW_NEG_THRESHOLD and cal_c > CAL_POS_THRESHOLD:
            flag = "SIGN-INVERTED"
            note = "calibration flipped a clearly negative raw correlation into a positive one"
        elif cal_c < 0 and raw_c > 0:
            flag = "REVERSED"
            note = "calibrated correlation went NEGATIVE despite raw agreement"
        elif raw_c < RAW_WEAK_THRESHOLD:
            flag = "weak-raw"
            note = f"raw correlation {raw_c:+.3f} < {RAW_WEAK_THRESHOLD}"
    if raw_c is None:
        polarity = "n/a"
    elif raw_c < RAW_NEG_THRESHOLD:
        polarity = "inverted"
    elif raw_c > CAL_POS_THRESHOLD:
        polarity = "positive"
    else:
        polarity = "no-signal"
    if raw_c is None or cal_c is None:
        stability = "n/a"
    elif abs(raw_c) <= POLARITY_EPS:
        stability = "stable"
    elif (raw_c > 0.0) != (cal_c > 0.0):
        stability = "flipped"
    else:
        stability = "stable"
    return {"flag": flag, "note": note, "polarity": polarity, "stability": stability}


def jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = {"array_shape": list(v.shape), "finite": int(np.isfinite(v).sum())}
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def mean(values) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def median(values) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


# ---------------------------------------------------------------------------
# tile discovery + inventory
# ---------------------------------------------------------------------------
class Tile:
    def __init__(self, tile_id: str, stratum: str, rgb: Path, dem: Path):
        self.tile_id = tile_id
        self.stratum = stratum
        self.rgb = rgb
        self.dem = dem
        # filled by inspect()
        self.info: dict = {}
        self.region: Optional[str] = None
        self.protocol_b_split: Optional[str] = None
        self.finetuned_split: Optional[str] = None
        # filled by _run_tile_inference
        self.reused_prediction = False
        self.prediction_provenance: Optional[str] = None
        self.reuse_note: Optional[str] = None

    def __repr__(self):
        return f"Tile({self.tile_id})"


def scan_tiles(root: Path, tiles: Optional[list] = None, strata: Optional[list] = None,
               limit: Optional[int] = None) -> tuple:
    warnings: list = []
    found = []
    for rgb in sorted(root.rglob("*_rgb.*")):
        if not rgb.stem.endswith("_rgb"):
            continue
        tile_id = rgb.stem[: -len("_rgb")]
        dem = rgb.parent / f"{tile_id}_dem.tif"
        if not dem.exists():
            warnings.append(f"missing DEM side for {rgb}")
            continue
        stratum = rgb.parent.name
        if stratum == root.name:
            stratum = "root"
        found.append(Tile(tile_id, stratum, rgb, dem))
    for dem in sorted(root.rglob("*_dem.tif")):
        tie = dem.with_name(dem.name.replace("_dem.tif", "_rgb.tif"))
        if not tie.exists():
            warnings.append(f"missing RGB side for {dem}")
    found.sort(key=lambda t: (t.stratum, t.tile_id))
    if strata:
        found = [t for t in found if t.stratum in strata]
    if tiles:
        want = set(tiles)
        found = [t for t in found if t.tile_id in want]
        missing = want - {t.tile_id for t in found}
        if missing:
            raise SystemExit(f"--tiles not found under {root}: {sorted(missing)}")
    if limit:
        found = found[:limit]
    return found, warnings


def _load_finetuned_val_regions(ckpt: Path) -> Optional[list]:
    if not ckpt.exists():
        return None
    import torch

    try:
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(str(ckpt), map_location="cpu")
    return payload.get("val_regions")


def inspect_tile(tile: Tile) -> None:
    import rasterio
    from rasterio.warp import transform_bounds

    info: dict = {}
    with rasterio.open(tile.rgb) as src:
        info["width"], info["height"] = src.width, src.height
        info["rgb_crs"] = str(src.crs) if src.crs else None
        info["rgb_transform"] = tuple(src.transform)[:6]
        info["rgb_bounds"] = tuple(src.bounds)
        info["rgb_count"] = src.count
    with rasterio.open(tile.dem) as src:
        info["dem_width"], info["dem_height"] = src.width, src.height
        info["dem_crs"] = str(src.crs) if src.crs else None
        info["dem_transform"] = tuple(src.transform)[:6]
        info["dem_bounds"] = tuple(src.bounds)
        info["nodata"] = src.nodata
        dem = src.read(1).astype("float64")
        nodata = src.nodata
        valid = np.isfinite(dem)
        if nodata is not None and np.isfinite(nodata):
            valid &= dem != nodata
        info["valid_pixels"] = int(valid.sum())
        info["total_pixels"] = int(dem.size)
        if info["valid_pixels"]:
            info["dem_min_m"] = float(dem[valid].min())
            info["dem_max_m"] = float(dem[valid].max())
        bounds = info["dem_bounds"]
        crs = info["dem_crs"] or "EPSG:4326"
        info["bounds_wgs84"] = tuple(round(v, 6) for v in transform_bounds(crs, "EPSG:4326", *bounds))
    info["rgb_sha256"] = sha256_file(tile.rgb)
    info["dem_sha256"] = sha256_file(tile.dem)
    info["aligned"] = "direct" if (
        info["width"] == info["dem_width"]
        and info["height"] == info["dem_height"]
        and (info["rgb_crs"] is None or info["dem_crs"] is None
             or (info["rgb_crs"] == info["dem_crs"]
                 and np.allclose(info["rgb_transform"], info["dem_transform"], atol=1e-9)))
    ) else "resampled"
    tile.info = info
    tile.region = assign_region(info["bounds_wgs84"])
    tile.protocol_b_split = DEFAULT_PROTOCOL_B_SPLIT.get(tile.region or "", "unassigned")


INVENTORY_COLUMNS = [
    "tile_id", "stratum", "region", "protocol_b_split", "finetuned_split",
    "rgb_path", "dem_path", "rgb_sha256", "dem_sha256", "aligned",
    "width", "height", "crs", "nodata", "valid_pixels", "total_pixels",
    "bounds_wgs84",
]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def inventory_row(tile: Tile) -> dict:
    i = tile.info
    return {
        "tile_id": tile.tile_id,
        "stratum": tile.stratum,
        "region": tile.region or "unassigned",
        "protocol_b_split": tile.protocol_b_split,
        "finetuned_split": tile.finetuned_split or "unknown",
        "rgb_path": _rel(tile.rgb),
        "dem_path": _rel(tile.dem),
        "rgb_sha256": i["rgb_sha256"],
        "dem_sha256": i["dem_sha256"],
        "aligned": i["aligned"],
        "width": i["width"],
        "height": i["height"],
        "crs": i["dem_crs"] or i["rgb_crs"],
        "nodata": i["nodata"],
        "valid_pixels": i["valid_pixels"],
        "total_pixels": i["total_pixels"],
        "bounds_wgs84": ",".join(f"{v:.6f}" for v in i["bounds_wgs84"]),
    }


def build_inventory(root: Path, ckpt: Path, tiles: Optional[list] = None,
                    strata: Optional[list] = None):
    found, warnings = scan_tiles(root, tiles=tiles, strata=strata)
    val_regions = _load_finetuned_val_regions(ckpt)
    for t in found:
        inspect_tile(t)
        if val_regions is not None:
            t.finetuned_split = "val" if t.tile_id in val_regions else "train"
    seen_sha: dict = {}
    rows = []
    for t in found:
        row = inventory_row(t)
        dup = seen_sha.get(row["dem_sha256"])
        if dup:
            warnings.append(f"duplicate DEM content: {t.tile_id} == {dup}")
        else:
            seen_sha[row["dem_sha256"]] = t.tile_id
        rows.append(row)
    return found, rows, warnings


def cmd_inventory(args) -> int:
    ckpt = Path(args.ckpt) if args.ckpt else DEFAULT_FINETUNED_CKPT
    found, rows, warnings = build_inventory(Path(args.root), ckpt, strata=args.stratum)
    out = Path(args.out) if args.out else MANIFEST_TEMPLATE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for warn in warnings:
        print(f"warning: {warn}", file=sys.stderr)
    regions: dict = {}
    for r in rows:
        regions.setdefault((r["region"], r["protocol_b_split"]), 0)
        regions[(r["region"], r["protocol_b_split"])] += 1
    print(f"inventory: {len(rows)} tiles -> {out}")
    for (region, split), n in sorted(regions.items()):
        print(f"  {region:<24} {split:<10} {n}")
    return 0


# ---------------------------------------------------------------------------
# model / env metadata
# ---------------------------------------------------------------------------
# Trivial control arms: prediction = image statistic, no model, no checkpoint.
# They bound how much of a score raw intensity structure alone explains.
_CONTROL_BACKENDS = ("brightness", "sobel")


def _control_infer(kind: str):
    def infer(rgb_hwc: np.ndarray) -> np.ndarray:
        gray = np.asarray(rgb_hwc, dtype="float64").mean(axis=2)
        if kind == "brightness":
            out = gray
        else:  # sobel: gradient magnitude of grayscale
            gy, gx = np.gradient(gray)
            out = np.hypot(gx, gy)
        lo, hi = float(out.min()), float(out.max())
        return ((out - lo) / (hi - lo)) if hi > lo else np.zeros_like(out)

    return infer


def _da3_map() -> dict:
    """{slug: (config attribute, default HF id)} for the opt-in DA3 backends."""
    from app.pipeline.depth import DA3_BACKENDS

    return {slug: (attr, default) for slug, attr, default in DA3_BACKENDS}


def _rs_map() -> dict:
    """{slug: (config attr, filename, license_note)} for the RS height backends."""
    from app.pipeline.depth import RS_BACKENDS

    return {slug: (attr, filename, note)
            for slug, attr, _repo, filename, note in RS_BACKENDS}


def _prediction_suffix(identity: dict) -> str:
    """Prediction file suffix: metric nDSM arms save metres, not rDSM."""
    return ("_ndsm" if identity["model"].get("output_kind") == "ndsm_metres"
            else "_relative")


def _force_backend(backend: str, model_path: Optional[str]) -> None:
    from app import config

    if backend == "auto" or backend in _CONTROL_BACKENDS:
        return
    config.FINETUNED_MODEL_PATH = ""
    config.IMELE_MODEL_PATH = ""
    for attr, _default in _da3_map().values():
        setattr(config, attr, "")
    for attr, _filename, _note in _rs_map().values():
        setattr(config, attr, "")
    if backend == "pix2pix":
        if not config.TF_MODEL_PATH:
            raise SystemExit("--backend pix2pix: no SavedModel found (models/pix2pix missing?)")
    elif backend == "imele":
        p = Path(model_path) if model_path else DEFAULT_IMELE_MODEL
        if not p.exists():
            raise SystemExit(f"--backend imele: model not found: {p}")
        config.IMELE_MODEL_PATH = str(p)
        config.TF_MODEL_PATH = ""
    elif backend == "finetuned":
        p = Path(model_path) if model_path else DEFAULT_FINETUNED_CKPT
        if not p.exists():
            raise SystemExit(f"--backend finetuned: checkpoint not found: {p}")
        config.FINETUNED_MODEL_PATH = str(p)
        config.TF_MODEL_PATH = ""
    elif backend == "depth_anything":
        config.TF_MODEL_PATH = ""
    elif backend in _da3_map():
        attr, default = _da3_map()[backend]
        setattr(config, attr, model_path or default)
        config.TF_MODEL_PATH = ""
    elif backend in _rs_map():
        attr, _filename, _note = _rs_map()[backend]
        # model_path overrides the default HF repo id (local .pth / snapshot dir)
        setattr(config, attr, model_path or _rs_default(backend))
        config.TF_MODEL_PATH = ""
    else:
        raise SystemExit(f"unknown backend: {backend}")


def _rs_default(slug: str) -> str:
    from app.pipeline.depth import RS_BACKENDS

    return next(repo for s, _attr, repo, _fn, _note in RS_BACKENDS if s == slug)


def _model_info(backend: str) -> dict:
    if backend in _CONTROL_BACKENDS:
        # no checkpoint bytes; the arm is fully pinned by its code (inference
        # fingerprint) and slug
        return {"slug": backend, "identifier": f"control:{backend}",
                "file_sha256": {}}

    from app.pipeline import depth

    slug = depth.backend_slug()
    expected = {"pix2pix": "pix2pix", "imele": "imele", "finetuned": "finetuned",
                "depth_anything": "depth_anything"}
    expected.update({s: s for s in _da3_map()})
    expected.update({s: s for s in _rs_map()})
    expected = expected.get(backend)
    if expected and slug != expected:
        raise RuntimeError(f"backend mismatch: asked for {backend}, loaded {slug}")
    ident = depth.model_identifier()
    if backend in _da3_map():
        # hub revision + weight sha256: a DA3 run names the exact bytes it used
        prov = depth.da3_checkpoint_provenance(ident)
        return {"slug": slug, "identifier": ident,
                "file_sha256": {prov["weight_file"]: prov["weight_sha256"]},
                "hf_repo_id": prov["repo_id"], "hf_revision": prov["revision"]}
    if backend in _rs_map():
        # metric RS height checkpoint: hub revision + weight sha256 + licence
        # note (altimap v2 is non-commercial - the flag must reach the run)
        # + compute settings (device, VRAM cap) + pinned DINOv2 backbone code
        _attr, filename, note = _rs_map()[backend]
        prov = depth.rs_checkpoint_provenance(ident, filename, note)
        return {"slug": slug, "identifier": ident,
                "output_kind": "ndsm_metres",
                "file_sha256": {prov["weight_file"]: prov["weight_sha256"]},
                "hf_repo_id": prov["repo_id"], "hf_revision": prov["revision"],
                "license_note": prov["license_note"],
                "compute": prov["compute"],
                "backbone_dinov2": prov["backbone_dinov2"]}
    hashes = {}
    p = Path(ident)
    if p.is_file():
        hashes[p.name] = sha256_file(p)
    elif p.is_dir():
        for sub in sorted(p.rglob("*")):
            if sub.is_file() and sub.suffix in (".pb", ".index", ".data-00000-of-00001",
                                                ".data-00000-of-00002", ".data-00001-of-00002"):
                hashes[_rel(sub)] = sha256_file(sub)
    return {"slug": slug, "identifier": ident, "file_sha256": hashes}


def _environment_info(seed: int) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "seed": seed,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for pkg in ("numpy", "scipy", "scikit-learn", "rasterio", "torch", "tensorflow", "pillow"):
        version = None
        try:
            version = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            try:
                version = getattr(importlib.import_module(pkg), "__version__", None)
            except Exception:  # noqa: BLE001
                version = None
        info[pkg] = version
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            # process-lifetime peak (one model per process): the run's VRAM cost
            info["cuda_max_memory_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1e6, 1)
        info["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except Exception:  # noqa: BLE001
        pass
    info["backend_env"] = {k: v for k, v in os.environ.items() if k.startswith("DEPTHWIZARD_")}
    return info


def _git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=15).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


# ---------------------------------------------------------------------------
# inference + per-tile scoring
# ---------------------------------------------------------------------------
def _load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            data = src.read().astype("float32")
    else:  # pragma: no cover - training tiles are GeoTIFFs
        from PIL import Image

        data = np.asarray(Image.open(path).convert("RGB"), dtype="float32").transpose(2, 0, 1)
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    return np.clip(data[:3].transpose(1, 2, 0), 0, 255).astype("uint8")


def _load_dem(path: Path) -> tuple:
    import rasterio

    with rasterio.open(path) as src:
        dem = src.read(1).astype("float64")
        meta = {"nodata": src.nodata, "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform)[:6], "bounds": tuple(src.bounds),
                "shape": (src.height, src.width)}
    return dem, meta


def _align_pair(rgb: np.ndarray, dem: np.ndarray, dem_meta: dict,
                rgb_meta: Optional[dict]) -> np.ndarray:
    if dem.shape == rgb.shape[:2]:
        return dem
    from app.pipeline.evaluation import align_dem_to_prediction
    from rasterio.transform import Affine

    if rgb_meta is None or rgb_meta.get("transform") is None or rgb_meta.get("crs") is None:
        raise ValueError(f"DEM shape {dem.shape} != RGB shape {rgb.shape[:2]} and no RGB georef to resample onto")
    return align_dem_to_prediction(
        np.zeros(rgb.shape[:2], dtype="float32"), dem,
        predicted_transform=Affine(*rgb_meta["transform"]), predicted_crs=rgb_meta["crs"],
        dem_transform=Affine(*dem_meta["transform"]), dem_crs=dem_meta["crs"])


def _rgb_meta(path: Path) -> Optional[dict]:
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None
    import rasterio

    with rasterio.open(path) as src:
        return {"crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform)[:6], "bounds": tuple(src.bounds)}


_TILE_GSD_M = 1.0  # frozen benchmark tiles are the 1 m Swiss grids


def _default_infer(rgb_hwc: np.ndarray) -> np.ndarray:
    from app.pipeline import depth

    if depth.active_output_kind() == "ndsm_metres":
        # gsd-aware: altimap resamples to its 0.33 m/px target and back;
        # RS3DAda (0.05-1.0 m census) ignores it and runs native.
        return depth.infer_ndsm_metres(rgb_hwc, gsd_in=_TILE_GSD_M)
    return depth.infer_relative_dsm(rgb_hwc)


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def _subsample_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


# ---------------------------------------------------------------------------
# prediction cache identity (sidecar JSON next to each .npy)
# ---------------------------------------------------------------------------
def _cache_sidecar(npy_path: Path) -> Path:
    return npy_path.with_suffix(".json")


def _write_prediction(run_dir: Path, tile: Tile, rdsm: np.ndarray, identity: dict) -> Path:
    """Save predictions/<tile>_relative.npy plus its identity sidecar.

    Metric RS height backends save <tile>_ndsm.npy instead: their array is
    nDSM in metres, not 0..1 rDSM, and evaluation.py --mode ndsm/dsm reads the
    two suffixes with exactly those semantics.

    The sidecar pins the inputs (rgb/dem sha256 + array shape/dtype), the model
    (backend slug, identifier, file hashes), the seed and the inference-code
    fingerprint that produced this array. A later --reuse-predictions run only
    accepts the cache when every one of those still matches.
    """
    p = run_dir / "predictions" / f"{tile.tile_id}{_prediction_suffix(identity)}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, rdsm.astype("float32"))
    payload = {
        "tile_id": tile.tile_id,
        "rgb_sha256": tile.info["rgb_sha256"],
        "dem_sha256": tile.info["dem_sha256"],
        "shape": list(rdsm.shape),
        "dtype": "float32",
        "prediction_sha256": sha256_file(p),
        "backend": identity["backend"],
        "model_identifier": identity["model"]["identifier"],
        "model_file_sha256": identity["model"]["file_sha256"],
        "seed": identity["seed"],
        "inference_fingerprint": identity["inference_fingerprint"],
        "provenance": tile.prediction_provenance or "inference",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _cache_sidecar(p).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _validate_cached_prediction(tile: Tile, dem: np.ndarray, reuse_dir: Path,
                                identity: dict, allow_unverified: bool) -> tuple:
    """Return (rdsm|None, provenance|None, reason|None) for a cache candidate.

    rdsm is set only when the cache is trustworthy: file present, array matches
    the tile grid, sidecar present and every pinned value (inputs, model, seed,
    inference code, file content) unchanged. A sidecar-less prediction is
    "legacy_unverified" and accepted only with allow_unverified=True. Anything
    rejected -> reason explains the miss and the caller re-runs inference.
    """
    cached = reuse_dir / f"{tile.tile_id}{_prediction_suffix(identity)}.npy"
    if not cached.is_file():
        return None, None, "not cached"
    rdsm = np.load(cached)
    if rdsm.shape != dem.shape:
        return None, None, f"cached shape {rdsm.shape} != tile grid {dem.shape}"
    sidecar = _cache_sidecar(cached)
    if not sidecar.is_file():
        if allow_unverified:
            return rdsm.astype("float64"), "legacy_unverified", None
        return None, None, ("no cache metadata (prediction predates identity sidecars); "
                            "pass --allow-unverified-reuse to accept it")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, "unreadable cache metadata sidecar"
    if (meta.get("rgb_sha256") != tile.info["rgb_sha256"]
            or meta.get("dem_sha256") != tile.info["dem_sha256"]):
        return None, None, "tile inputs changed since cache was written (rgb/dem sha256 mismatch)"
    if (meta.get("backend") != identity["backend"]
            or meta.get("model_identifier") != identity["model"]["identifier"]
            or meta.get("model_file_sha256") != identity["model"]["file_sha256"]):
        return None, None, (f"model changed since cache was written (cached "
                            f"{meta.get('backend')} / {meta.get('model_identifier')})")
    if meta.get("seed") != identity["seed"]:
        return None, None, "seed changed since cache was written"
    if meta.get("inference_fingerprint") != identity["inference_fingerprint"]:
        return None, None, "inference code changed since cache was written"
    if (meta.get("shape") != list(rdsm.shape) or meta.get("dtype") != str(rdsm.dtype)):
        return None, None, "cached array shape/dtype does not match its metadata"
    if sha256_file(cached) != meta.get("prediction_sha256"):
        return None, None, "cached prediction file content changed (sha256 mismatch)"
    return rdsm.astype("float64"), "reuse_verified", None


class TilePrediction:
    """Everything one evaluated tile produced, JSON/CSV-ready."""

    def __init__(self, tile: Tile):
        self.tile = tile
        self.row: dict = {"tile_id": tile.tile_id, "stratum": tile.stratum,
                          "region": tile.region or "unassigned",
                          "protocol_b_split": tile.protocol_b_split or "",
                          "raw_corr": None, "cal_corr": None, "rmse": None, "mae": None,
                          "held_out": None, "valid_pixels": None,
                          "calibration_scale": None, "calibration_offset": None,
                          "calibration_status": None, "calibration_warning": None,
                          "raw_correlation_signed": None,
                          "calibration_ransac_scale": None, "calibration_ols_scale": None,
                          "baseline_rmse": None, "baseline_mae": None,
                          "flag": None, "polarity": None, "stability": None,
                          "note": None, "error": None}
        self.detail: dict = {}


def _score_historical(tile: Tile, rdsm: np.ndarray, dem: np.ndarray,
                      nodata, infer_seconds: float) -> TilePrediction:
    from app.pipeline.evaluation import evaluate_prediction_truth

    out = TilePrediction(tile)
    rel = evaluate_prediction_truth(rdsm, dem, nodata=nodata, mode="relative")
    cal = evaluate_prediction_truth(rdsm, dem, nodata=nodata, mode="calibrated")
    v = verdict(rel["correlation"], cal["correlation"])
    out.row.update({
        "raw_corr": rel["correlation"], "cal_corr": cal["correlation"],
        "rmse": cal["rmse"], "mae": cal["mae"],
        "held_out": cal["held_out_pixel_count"], "valid_pixels": rel["valid_pixels"],
        "calibration_scale": cal["calibration_scale"],
        "calibration_offset": cal["calibration_offset"],
        "calibration_status": cal["calibration_status"],
        "calibration_warning": cal["calibration_warning"],
        "raw_correlation_signed": cal["raw_correlation_signed"],
        "calibration_ransac_scale": cal["calibration_ransac_scale"],
        "calibration_ols_scale": cal["calibration_ols_scale"],
        "flag": v["flag"], "polarity": v["polarity"], "stability": v["stability"],
        "note": v["note"],
    })
    out.detail = {
        "relative_mode": jsonable(rel),
        "calibrated_mode": jsonable(cal),
        "verdict": v,
        "infer_seconds": round(infer_seconds, 3),
        "prediction_reused": bool(tile.reused_prediction),
        "prediction_provenance": tile.prediction_provenance,
        "reuse_note": tile.reuse_note,
        "predicted_elevation": (cal.get("predicted_elevation")
                                if isinstance(cal.get("predicted_elevation"), np.ndarray) else None),
    }
    return out


def _pool_pixels(rdsm: np.ndarray, dem: np.ndarray, nodata, cap: int) -> tuple:
    from app.pipeline.evaluation import build_valid_mask

    valid = build_valid_mask(rdsm, dem, nodata=nodata)
    d = rdsm[valid].astype("float64")
    h = dem[valid].astype("float64")
    idx = _subsample_indices(d.size, cap)
    return d[idx], h[idx], int(valid.sum())


def _score_frozen(tile: Tile, rdsm: np.ndarray, dem: np.ndarray, nodata,
                  fit, baseline_value: float, infer_seconds: float) -> TilePrediction:
    from app.pipeline.evaluation import build_valid_mask
    from app.pipeline.validate import compute_metrics, correlation_and_reason
    from app.pipeline.calibration import apply_fit

    out = TilePrediction(tile)
    valid = build_valid_mask(rdsm, dem, nodata=nodata)
    if not valid.any():
        raise ValueError("no valid pixels")
    d = rdsm[valid].astype("float64")
    h = dem[valid].astype("float64")
    pred_m = apply_fit(d, fit)
    metrics = compute_metrics(pred_m, h)
    raw_c, raw_reason = correlation_and_reason(d, h)
    cal_c, cal_reason = correlation_and_reason(pred_m, h)
    base_metrics = compute_metrics(np.full_like(h, baseline_value), h)
    base_c, base_reason = correlation_and_reason(np.full_like(h, baseline_value), h)
    v = verdict(raw_c, cal_c)
    out.row.update({
        "raw_corr": raw_c, "cal_corr": cal_c, "rmse": metrics["rmse"], "mae": metrics["mae"],
        "held_out": None, "valid_pixels": int(valid.sum()),
        "calibration_scale": fit.scale, "calibration_offset": fit.offset,
        "calibration_status": fit.calibration_status,
        "calibration_warning": fit.calibration_warning,
        "raw_correlation_signed": raw_c,
        "calibration_ransac_scale": fit.ransac_scale, "calibration_ols_scale": fit.ols_scale,
        "baseline_rmse": base_metrics["rmse"], "baseline_mae": base_metrics["mae"],
        "flag": v["flag"], "polarity": v["polarity"], "stability": v["stability"],
        "note": v["note"],
    })
    predicted_elevation = np.where(valid, pred_m.reshape(rdsm.shape), np.nan)
    out.detail = {
        "metrics_all_valid_pixels": {**metrics, "correlation": cal_c, "correlation_reason": cal_reason},
        "raw": {"correlation": raw_c, "correlation_reason": raw_reason},
        "baseline_constant": {**base_metrics, "correlation": base_c,
                              "correlation_reason": base_reason,
                              "value_m": baseline_value},
        "verdict": v,
        "infer_seconds": round(infer_seconds, 3),
        "prediction_reused": bool(tile.reused_prediction),
        "prediction_provenance": tile.prediction_provenance,
        "reuse_note": tile.reuse_note,
        "predicted_elevation": predicted_elevation,
    }
    return out


def _save_tile_outputs(run_dir: Path, tile: Tile, rdsm: np.ndarray,
                       outcome: TilePrediction, rgb: np.ndarray, dem: np.ndarray,
                       save_predictions: bool, save_comparisons: bool,
                       identity: dict) -> None:
    detail = dict(outcome.detail)
    predicted = detail.pop("predicted_elevation", None)
    detail["outputs"] = {}
    if save_predictions:
        p = _write_prediction(run_dir, tile, rdsm, identity)
        detail["outputs"]["prediction_npy"] = _rel(p)
    if save_comparisons:
        p = run_dir / "comparisons" / f"{tile.tile_id}.png"
        if predicted is not None:
            from app.pipeline.evaluation import render_height_comparison

            render_height_comparison(rgb, dem, predicted, p)
            detail["outputs"]["comparison_png"] = _rel(p)
        else:
            detail["outputs"]["comparison_png"] = None
            detail["outputs"]["comparison_skipped"] = "no calibrated prediction for this tile"
    tdir = run_dir / "per_tile"
    tdir.mkdir(parents=True, exist_ok=True)
    payload = {"tile_id": tile.tile_id, "row": jsonable(outcome.row), **jsonable(detail)}
    (tdir / f"{tile.tile_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------
def _run_tile_inference(tile: Tile, infer_fn: Callable, seed: int,
                        reuse_dir: Optional[str], identity: dict,
                        allow_unverified_reuse: bool = False) -> tuple:
    _apply_seed(seed)
    rgb = _load_rgb(tile.rgb)
    dem, dem_meta = _load_dem(tile.dem)
    dem = _align_pair(rgb, dem, dem_meta, _rgb_meta(tile.rgb))
    tile.reused_prediction = False
    tile.prediction_provenance = None
    tile.reuse_note = None
    if reuse_dir:
        rdsm, provenance, reason = _validate_cached_prediction(
            tile, dem, Path(reuse_dir), identity, allow_unverified_reuse)
        if rdsm is not None:
            tile.reused_prediction = True
            tile.prediction_provenance = provenance
            return rgb, dem, dem_meta["nodata"], rdsm, 0.0
        tile.reuse_note = reason
        print(f"  reuse miss {tile.tile_id}: {reason}; re-running inference", flush=True)
    t0 = time.time()
    rdsm = infer_fn(rgb)
    tile.prediction_provenance = "inference"
    return rgb, dem, dem_meta["nodata"], np.asarray(rdsm, dtype="float64"), time.time() - t0


def run_historical(tiles: list, infer_fn: Callable, run_dir: Path, args,
                   identity: dict) -> dict:
    rows, errors = [], []
    for tile in tiles:
        try:
            rgb, dem, nodata, rdsm, secs = _run_tile_inference(
                tile, infer_fn, args.seed, args.reuse_predictions, identity,
                args.allow_unverified_reuse)
            outcome = _score_historical(tile, rdsm, dem, nodata, secs)
            _save_tile_outputs(run_dir, tile, rdsm, outcome, rgb, dem,
                               args.save_predictions, args.save_comparisons, identity)
            rows.append(outcome.row)
        except Exception as e:  # noqa: BLE001
            rows.append({"tile_id": tile.tile_id, "stratum": tile.stratum,
                         "region": tile.region or "unassigned", "error": f"{type(e).__name__}: {e}"})
            errors.append(f"{tile.tile_id}: {type(e).__name__}: {e}")
    groups = {}
    for r in rows:
        if r.get("error"):
            continue
        g = groups.setdefault(r["stratum"], [])
        g.append(r)
    aggregates = {}
    for g, items in sorted(groups.items()):
        aggregates[g] = {
            "n_tiles": len(items),
            "mean_raw_corr": mean([i["raw_corr"] for i in items]),
            "mean_cal_corr": mean([i["cal_corr"] for i in items]),
            "mean_rmse": mean([i["rmse"] for i in items]),
            "mean_mae": mean([i["mae"] for i in items]),
        }
    overall = {k: mean([r[k] for r in rows if not r.get("error")])
               for k in ("raw_corr", "cal_corr", "rmse", "mae")}
    flags = {}
    for r in rows:
        if r.get("flag"):
            flags[r["flag"]] = flags.get(r["flag"], 0) + 1
    return {"rows": rows, "errors": errors, "aggregates": aggregates,
            "overall": overall, "flags": flags}


def _load_split_manifest(path: Path) -> dict:
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tile_id"):
                mapping[row["tile_id"]] = row
    return mapping


def _apply_manifest(tiles: list, manifest: Optional[Path]) -> None:
    if not manifest:
        return
    mapping = _load_split_manifest(manifest)
    for t in tiles:
        row = mapping.get(t.tile_id)
        if row:
            t.region = row.get("region") or t.region
            t.protocol_b_split = row.get("protocol_b_split") or t.protocol_b_split


def run_geographic(tiles: list, infer_fn: Callable, run_dir: Path, args,
                   identity: dict) -> dict:
    from app.pipeline.calibration import fit_affine

    dev = [t for t in tiles if t.protocol_b_split == "dev"]
    test = [t for t in tiles if t.protocol_b_split == "test"]
    skipped = [t for t in tiles if t.protocol_b_split not in ("dev", "test")]
    if not dev or not test:
        raise SystemExit(f"geographic protocol needs dev and test tiles; got {len(dev)} dev, "
                         f"{len(test)} test, {len(skipped)} unassigned")
    rows, errors = [], []
    pooled_d, pooled_h = [], []
    per_tile_cap = max(1, args.calibration_pixel_cap // len(dev))
    for tile in dev:
        try:
            rgb, dem, nodata, rdsm, secs = _run_tile_inference(
                tile, infer_fn, args.seed, args.reuse_predictions, identity,
                args.allow_unverified_reuse)
            d, h, n_valid = _pool_pixels(rdsm, dem, nodata, per_tile_cap)
            pooled_d.append(d)
            pooled_h.append(h)
            if args.save_predictions:
                _write_prediction(run_dir, tile, rdsm, identity)
            # dev tiles are calibration inputs; no calibrated prediction to render
            (run_dir / "per_tile").mkdir(parents=True, exist_ok=True)
            (run_dir / "per_tile" / f"{tile.tile_id}.json").write_text(json.dumps({
                "tile_id": tile.tile_id, "split": "dev", "region": tile.region,
                "valid_pixels": n_valid, "pixels_pooled": int(d.size),
                "infer_seconds": round(secs, 3),
                "prediction_reused": bool(tile.reused_prediction),
                "prediction_provenance": tile.prediction_provenance,
                "reuse_note": tile.reuse_note}, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            errors.append(f"dev {tile.tile_id}: {type(e).__name__}: {e}")
    if not pooled_d:
        raise SystemExit("geographic protocol: no dev tile could be inferred")
    pooled_d = np.concatenate(pooled_d)
    pooled_h = np.concatenate(pooled_h)
    fit = fit_affine(pooled_d, pooled_h)
    baseline_value = float(np.median(pooled_h))
    for tile in test:
        try:
            rgb, dem, nodata, rdsm, secs = _run_tile_inference(
                tile, infer_fn, args.seed, args.reuse_predictions, identity,
                args.allow_unverified_reuse)
            outcome = _score_frozen(tile, rdsm, dem, nodata, fit, baseline_value, secs)
            _save_tile_outputs(run_dir, tile, rdsm, outcome, rgb, dem,
                               args.save_predictions, args.save_comparisons, identity)
            rows.append(outcome.row)
        except Exception as e:  # noqa: BLE001
            rows.append({"tile_id": tile.tile_id, "region": tile.region or "unassigned",
                         "error": f"{type(e).__name__}: {e}"})
            errors.append(f"test {tile.tile_id}: {type(e).__name__}: {e}")
    from app.pipeline.calibration import degenerate_reason

    calibration = {
        "fit_on": "pooled dev-region pixels only",
        "dev_regions": sorted({t.region for t in dev}),
        "test_regions": sorted({t.region for t in test}),
        "dev_tiles": [t.tile_id for t in dev],
        "test_tiles": [t.tile_id for t in test],
        "unassigned_tiles": [t.tile_id for t in skipped],
        "pooled_pixels": int(pooled_d.size),
        "pixel_cap": args.calibration_pixel_cap,
        "scale": fit.scale, "offset": fit.offset,
        "n_samples": fit.n_samples, "n_inliers": fit.n_inliers,
        "status": fit.calibration_status, "warning": fit.calibration_warning,
        "ransac_scale": fit.ransac_scale, "ols_scale": fit.ols_scale,
        "degenerate_reason": degenerate_reason(fit),
        "polarity_inverted": fit.polarity_inverted,
        "polarity_reason": fit.polarity_reason,
        "baseline_constant_m": baseline_value,
    }
    return {"rows": rows, "errors": errors, "calibration": calibration,
            "dev_rows": [], "group_by": "region"}


def run_local_calibration(tiles: list, infer_fn: Callable, run_dir: Path, args,
                          identity: dict) -> dict:
    from app.pipeline.calibration import fit_affine, apply_fit, degenerate_reason

    regions = {}
    for t in tiles:
        regions.setdefault(t.region or "unassigned", []).append(t)
    rows, errors, region_summaries = [], [], {}
    for region, rtiles in sorted(regions.items()):
        rtiles = sorted(rtiles, key=lambda t: (t.info["bounds_wgs84"][0], t.info["bounds_wgs84"][1]))
        n_calib = math.ceil(len(rtiles) / 2)
        calib_block, test_all = rtiles[:n_calib], rtiles[n_calib:]
        test_block = [t for t in test_all
                      if all(bbox_gap_deg(t.info["bounds_wgs84"], c.info["bounds_wgs84"])
                             >= args.local_buffer_deg - 1e-9
                             for c in calib_block)]
        buffered_out = [t.tile_id for t in test_all if t not in test_block]
        summary = {"region": region, "n_tiles": len(rtiles),
                   "calibration_block": [t.tile_id for t in calib_block],
                   "test_block": [t.tile_id for t in test_block],
                   "buffered_out": buffered_out,
                   "buffer_deg": args.local_buffer_deg}
        if not test_block:
            summary["skipped"] = "no test tiles spatially separated from the calibration block"
            region_summaries[region] = summary
            continue
        pooled_d, pooled_h = [], []
        cap = max(1, args.calibration_pixel_cap // len(calib_block))
        for tile in calib_block:
            try:
                rgb, dem, nodata, rdsm, secs = _run_tile_inference(
                    tile, infer_fn, args.seed, args.reuse_predictions, identity,
                    args.allow_unverified_reuse)
                d, h, _ = _pool_pixels(rdsm, dem, nodata, cap)
                pooled_d.append(d)
                pooled_h.append(h)
                if args.save_predictions:
                    _write_prediction(run_dir, tile, rdsm, identity)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{region} calib {tile.tile_id}: {type(e).__name__}: {e}")
        if not pooled_d:
            summary["skipped"] = "no calibration tile could be inferred"
            region_summaries[region] = summary
            continue
        fit = fit_affine(np.concatenate(pooled_d), np.concatenate(pooled_h))
        baseline_value = float(np.median(np.concatenate(pooled_h)))
        summary.update({"scale": fit.scale, "offset": fit.offset,
                        "status": fit.calibration_status, "warning": fit.calibration_warning,
                        "ransac_scale": fit.ransac_scale, "ols_scale": fit.ols_scale,
                        "degenerate_reason": degenerate_reason(fit),
                        "polarity_inverted": fit.polarity_inverted,
                        "pooled_pixels": int(sum(d.size for d in pooled_d)),
                        "baseline_constant_m": baseline_value})
        for tile in test_block:
            try:
                rgb, dem, nodata, rdsm, secs = _run_tile_inference(
                    tile, infer_fn, args.seed, args.reuse_predictions, identity,
                    args.allow_unverified_reuse)
                outcome = _score_frozen(tile, rdsm, dem, nodata, fit, baseline_value, secs)
                _save_tile_outputs(run_dir, tile, rdsm, outcome, rgb, dem,
                                   args.save_predictions, args.save_comparisons, identity)
                rows.append(outcome.row)
            except Exception as e:  # noqa: BLE001
                rows.append({"tile_id": tile.tile_id, "region": region,
                             "error": f"{type(e).__name__}: {e}"})
                errors.append(f"{region} test {tile.tile_id}: {type(e).__name__}: {e}")
        region_summaries[region] = summary
    return {"rows": rows, "errors": errors, "regions": region_summaries,
            "group_by": "region"}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _parse_reference(path: Path) -> list:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8", "utf-16", "utf-16-le"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise SystemExit(f"cannot decode reference file: {path}")
    listing, table, in_table = [], [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Picked up"):
            in_table = False
        elif "_rgb" in stripped and " + " in stripped and stripped.endswith((".tif", ".png")):
            listing.append(Path(stripped.split()[0]).stem[: -len("_rgb")])
        elif set(stripped) == {"="} and "====" in stripped:
            in_table = len(table) == 0 and bool(listing)
        elif in_table and stripped.startswith("(") is False:
            parts = stripped.split()
            if len(parts) == 7 and parts[0] not in ("tile",) and not parts[1].startswith("raw_"):
                table.append({
                    "tile": parts[0],
                    "raw_corr": None if parts[1] == "n/a" else float(parts[1]),
                    "cal_corr": None if parts[2] == "n/a" else float(parts[2]),
                    "rmse": None if parts[3] == "n/a" else float(parts[3]),
                    "mae": None if parts[4] == "n/a" else float(parts[4]),
                    "held_out": None if parts[5] == "n/a" else int(parts[5]),
                    "flag": parts[6],
                })
    if not table:
        raise SystemExit(f"reference table not found in {path}")
    if len(listing) == len(table):
        for tile_id, row in zip(listing, table):
            row["tile_id"] = tile_id
    else:
        for i, row in enumerate(table):
            row["tile_id"] = f"row_{i + 1}"
    return table


def compare_reference(rows: list, reference: Path, tol: float) -> dict:
    ref = _parse_reference(reference)
    by_id = {r["tile_id"]: r for r in ref}
    comparisons = []
    deltas_by_metric: dict = {"raw_corr": [], "cal_corr": [], "rmse": [], "mae": []}
    mismatches = []
    for row in rows:
        if row.get("error"):
            continue
        r = by_id.get(row["tile_id"])
        if r is None:
            mismatches.append(f"{row['tile_id']}: not in reference")
            continue
        entry = {"tile_id": row["tile_id"]}
        within = True
        for m in deltas_by_metric:
            new, old = row.get(m), r.get(m)
            if new is None or old is None:
                entry[m] = None
                entry[f"delta_{m}"] = None
                continue
            delta = float(new - old)
            entry[m] = new
            entry[f"ref_{m}"] = old
            entry[f"delta_{m}"] = delta
            deltas_by_metric[m].append(abs(delta))
            if abs(delta) > tol:
                within = False
        entry["flag"] = row.get("flag")
        entry["ref_flag"] = r.get("flag")
        entry["within_tol"] = within
        comparisons.append(entry)
        if not within:
            mismatches.append(f"{row['tile_id']}: delta exceeds tol={tol}")
    return {
        "reference": _rel(reference),
        "reference_tiles": len(ref),
        "compared_tiles": len(comparisons),
        "tolerance": tol,
        "n_within_tolerance": sum(1 for c in comparisons if c["within_tol"]),
        "max_abs_delta": {m: (max(v) if v else None) for m, v in deltas_by_metric.items()},
        "mismatches": mismatches,
        "per_tile": comparisons,
    }


def _group_aggregates(rows: list, group_by: str) -> dict:
    groups: dict = {}
    for r in rows:
        if r.get("error"):
            continue
        groups.setdefault(r.get(group_by) or "unassigned", []).append(r)
    out = {}
    for g, items in sorted(groups.items()):
        out[g] = {
            "n_tiles": len(items),
            "mean_raw_corr": mean([i.get("raw_corr") for i in items]),
            "mean_cal_corr": mean([i.get("cal_corr") for i in items]),
            "mean_rmse": mean([i.get("rmse") for i in items]),
            "mean_mae": mean([i.get("mae") for i in items]),
            "mean_baseline_rmse": mean([i.get("baseline_rmse") for i in items]),
            "mean_baseline_mae": mean([i.get("baseline_mae") for i in items]),
            "sum_valid_pixels": int(sum(i.get("valid_pixels") or 0 for i in items)),
        }
    return out


def _render_summary_md(summary: dict, run_dir: Path) -> str:
    lines = [f"# DepthWizard elevation benchmark - {summary['protocol']}", ""]
    run = summary["run"]
    lines += [
        f"- commit: `{run['git']['commit'][:12]}` (branch `{run['git']['branch']}`, dirty={run['git']['dirty']})",
        f"- backend: **{summary['backend']['slug']}** (`{summary['backend']['identifier']}`)",
        f"- started (UTC): {run['started_utc']}",
        f"- seed: {run['seed']}, calibration pixel cap: {run['calibration_pixel_cap']}",
        f"- tiles: {summary['tiles']['succeeded']}/{summary['tiles']['total']} scored"
        + (f", {len(summary['tiles']['errors'])} errors" if summary["tiles"]["errors"] else ""),
        f"- outputs: `{_rel(run_dir)}`",
        "",
    ]
    if summary["protocol"] == "historical":
        lines += [
            "Protocol A (legacy per-tile 80/20 PIXEL split): raw correlation uses ALL valid pixels;",
            "RMSE/MAE/calibrated correlation are on the held-out 20% of each tile's pixels.",
            "This is a reproduction/diagnostic of the historical scorer, not independent validation",
            "where tiles overlap the backend's training data.",
            "",
            "| stratum | tiles | raw corr | calibrated corr | RMSE (m) | MAE (m) |",
            "|---|---|---|---|---|---|",
        ]
        for g, a in summary["aggregates"].items():
            lines.append(f"| {g} | {a['n_tiles']} | {_fmt(a['mean_raw_corr'])} | "
                         f"{_fmt(a['mean_cal_corr'])} | {_fmt(a['mean_rmse'], 2)} | "
                         f"{_fmt(a['mean_mae'], 2)} |")
        o = summary["overall"]
        lines.append(f"| **all** | {summary['tiles']['succeeded']} | {_fmt(o['raw_corr'])} | "
                     f"{_fmt(o['cal_corr'])} | {_fmt(o['rmse'], 2)} | {_fmt(o['mae'], 2)} |")
        lines.append("")
        lines.append(f"Flags: {summary['flags']}")
        if summary.get("reference_comparison"):
            rc = summary["reference_comparison"]
            lines += [
                "",
                f"## Comparison vs `{rc['reference']}`",
                "",
                f"- compared tiles: {rc['compared_tiles']} (reference has {rc['reference_tiles']})",
                f"- within tolerance {rc['tolerance']}: {rc['n_within_tolerance']}/{rc['compared_tiles']}",
                f"- max |delta|: {rc['max_abs_delta']}",
            ]
            if rc["mismatches"]:
                lines.append(f"- mismatches: {rc['mismatches'][:10]}"
                             + (" ..." if len(rc["mismatches"]) > 10 else ""))
    else:
        if summary["protocol"] == "geographic":
            cal = summary["calibration"]
            lines += [
                f"Calibration fit on **dev regions only**: {', '.join(cal['dev_regions'])} "
                f"({len(cal['dev_tiles'])} tiles, {cal['pooled_pixels']} pooled pixels"
                f"{', capped at ' + str(cal['pixel_cap']) if cal['pooled_pixels'] >= cal['pixel_cap'] else ''}).",
                f"Frozen scale={_fmt(cal['scale'])}, offset={_fmt(cal['offset'], 2)} m, "
                f"status={cal['status']}, warning={cal['warning']}.",
                f"Constant baseline: dev median elevation = {_fmt(cal['baseline_constant_m'], 2)} m.",
                "",
            ]
        lines += [
            "| region | tiles | raw corr | calibrated corr | RMSE (m) | MAE (m) | baseline RMSE (m) | baseline MAE (m) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for g, a in summary["groups"].items():
            lines.append(f"| {g} | {a['n_tiles']} | {_fmt(a['mean_raw_corr'])} | "
                         f"{_fmt(a['mean_cal_corr'])} | {_fmt(a['mean_rmse'], 2)} | "
                         f"{_fmt(a['mean_mae'], 2)} | {_fmt(a['mean_baseline_rmse'], 2)} | "
                         f"{_fmt(a['mean_baseline_mae'], 2)} |")
        lines.append("")
        if summary["protocol"] == "local_calibration":
            lines += ["## Per-region calibration snapshots (calibration fit on one spatial block,",
                      "tested on a block at least one buffer width away)", "",
                      "| region | calib tiles | test tiles | buffered out | scale | offset | status |",
                      "|---|---|---|---|---|---|---|"]
            for g, s in summary["regions"].items():
                if s.get("skipped"):
                    lines.append(f"| {g} | {len(s['calibration_block'])} | 0 | "
                                 f"{len(s['buffered_out'])} | - | - | skipped: {s['skipped']} |")
                else:
                    lines.append(f"| {g} | {len(s['calibration_block'])} | {len(s['test_block'])} | "
                                 f"{len(s['buffered_out'])} | {_fmt(s['scale'])} | "
                                 f"{_fmt(s['offset'], 2)} | {s['status']} |")
        lines += ["", f"Flags: {summary['flags']}"]
    lines += ["", "## Limitations", ""]
    for lim in summary["limitations"]:
        lines.append(f"- {lim}")
    lines += ["", "## Environment", "", "```json",
              json.dumps(summary["environment"], indent=2), "```", ""]
    return "\n".join(lines)


def _fmt(v, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:+.{digits}f}" if digits == 3 else f"{v:.{digits}f}"


def _limitations(summary: dict) -> list:
    lim = [
        "Model inference can differ at float precision across hardware (GPU/CPU); "
        "environment.json records the exact stack used.",
    ]
    backend = summary["backend"]["slug"]
    overlap = summary.get("overlap", {})
    if backend == "pix2pix":
        lim.append("pix2pix (ImageToDEM Sentinel-2+ALOS provenance) has no documented overlap "
                   "with these tiles, but its external training data could not be audited from "
                   "this repository.")
    if backend == "finetuned":
        n_train = len(overlap.get("finetuned_train", []))
        n_val = len(overlap.get("finetuned_val", []))
        lim.append(f"finetuned decoder: {n_train} evaluated tiles are in its TRAINING data and "
                   f"{n_val} in its validation split (seed-0 region split) - results are a "
                   "diagnostic reproduction with training overlap, NOT independent validation.")
    if backend == "imele":
        lim.append("IMELE's training corpus is not documented in this repository; overlap with "
                   "these tiles cannot be ruled out.")
    if backend.startswith("da3"):
        lim.append(f"{backend} is a general-purpose monocular-depth checkpoint; overlap of its "
                   "training corpus with these tiles cannot be ruled out.")
        if backend in ("da3-large", "da3-giant"):
            lim.append(f"{backend} weights are CC BY-NC 4.0: non-commercial use only. "
                       "da3-giant is additionally hardware-skipped here (1.15B params do not fit "
                       "this box's 4 GB VRAM).")
    if backend in _CONTROL_BACKENDS:
        lim.append(f"{backend} control arm: predicts an image statistic (not depth); it bounds "
                   "how much of a score trivial intensity structure alone explains.")
    if backend in _rs_map() or summary["backend"].get("output_kind") == "ndsm_metres":
        note = (summary["backend"].get("license_note") or "").strip()
        lim.append(
            f"{backend} is a metric nDSM checkpoint: predictions/<tile>_ndsm.npy carries "
            "height above ground in METRES, not 0..1 rDSM. Any rDSM-protocol score in this "
            "run dir treats it as a relative surface and is NOT meaningful for this arm - "
            "score it with evaluation.py --mode ndsm/dsm instead."
            + (f" Licence: {note}." if note else ""))
    if summary["protocol"] == "historical":
        lim.append("Legacy per-tile split shares pixels of the SAME tile between fit and "
                   "score (80/20 pixel split); adjacent tiles also share scenery. Not a "
                   "geographic validation.")
    if summary["protocol"] == "geographic":
        lim.append("Frozen cross-region calibration transfers one affine scale/offset fitted on "
                   "dev regions to unseen regions; absolute elevation offsets differ between "
                   "regions, so RMSE includes a real region-bias term (that is the point of "
                   "this protocol).")
    if summary["protocol"] == "local_calibration":
        lim.append("Local-calibration test blocks are tile-level spatial separations within one "
                   "region; they still share the region's terrain character with the calibration "
                   "block (not a cross-region transfer test).")
    if summary.get("reference_comparison"):
        lim.append("The historical reference was produced 2026-09-24 by server code that predates "
                   "the current calibration logic (no low-relief guard, no RANSAC-vs-OLS sign "
                   "reconciliation). Where RANSAC and OLS slopes disagree in sign, the calibrated "
                   "correlation can legitimately differ in SIGN, and low-relief tiles can now "
                   "report a degenerate fit; see tools/diagnose_calibration.py for the per-tile "
                   "audit of the seven differing tiles.")
        lim.append("The historical run scored tiles over HTTP through the /evaluate route; this "
                   "benchmark runs inference in-process. Tiny numeric differences (TF version, "
                   "oneDNN kernels) can shift correlations by ~1e-3.")
    return lim


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    root = Path(args.root)
    tiles, warnings = scan_tiles(root, tiles=args.tiles, strata=args.stratum, limit=args.limit)
    for warn in warnings:
        print(f"warning: {warn}", file=sys.stderr)
    if not tiles:
        raise SystemExit(f"no tiles found under {root}")
    ckpt = Path(args.ckpt) if args.ckpt else DEFAULT_FINETUNED_CKPT
    val_regions = _load_finetuned_val_regions(ckpt)
    for t in tiles:
        inspect_tile(t)
        if val_regions is not None:
            t.finetuned_split = "val" if t.tile_id in val_regions else "train"
    _apply_manifest(tiles, Path(args.manifest) if args.manifest else None)
    _force_backend(args.backend, args.model_path)
    _apply_seed(args.seed)

    # Resolve the model identity BEFORE inference: --reuse-predictions validates
    # cache sidecars against it, and every saved prediction gets one written.
    model_info = _model_info(args.backend)
    identity = {
        "backend": model_info["slug"],
        "model": model_info,
        "seed": args.seed,
        "inference_fingerprint": _inference_fingerprint(),
    }

    # Load the model up front: fail fast when it cannot load, and measure the
    # one-time load separately from per-tile inference (model_load_seconds).
    model_load_seconds = 0.0
    if args.backend not in _CONTROL_BACKENDS:
        from app.pipeline import depth as depth_mod

        t_load = time.time()
        depth_mod.ensure_model()
        model_load_seconds = round(time.time() - t_load, 1)

    git = _git_info()
    backend_short = {"auto": "auto", "depth_anything": "da2"}.get(args.backend, args.backend)
    started = datetime.now(timezone.utc)
    base = f"{git['commit'][:7]}_{backend_short}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(args.run_root) / base
    suffix = 2
    while run_dir.exists():
        run_dir = Path(args.run_root) / f"{base}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    infer_fn = _control_infer(args.backend) if args.backend in _CONTROL_BACKENDS else _default_infer
    if args.backend in _CONTROL_BACKENDS:
        print(f"note: --backend {args.backend} is a control arm (image statistic, not depth)",
              file=sys.stderr)
    elif args.backend == "auto":
        print("note: --backend auto uses the config dispatch order (see AGENTS.md)", file=sys.stderr)

    print(f"run dir: {run_dir}")
    print(f"tiles: {len(tiles)}", flush=True)
    t0 = time.time()
    if args.protocol == "historical":
        protocol_out = run_historical(tiles, infer_fn, run_dir, args, identity)
    elif args.protocol == "geographic":
        protocol_out = run_geographic(tiles, infer_fn, run_dir, args, identity)
    else:
        protocol_out = run_local_calibration(tiles, infer_fn, run_dir, args, identity)
    elapsed = time.time() - t0

    rows = protocol_out["rows"]
    errors = protocol_out["errors"]
    succeeded = [r for r in rows if not r.get("error")]

    group_by = protocol_out.get("group_by", "stratum")
    groups = _group_aggregates(rows, group_by)
    flags: dict = {}
    for r in succeeded:
        if r.get("flag"):
            flags[r["flag"]] = flags.get(r["flag"], 0) + 1
    overlap = {
        "finetuned_val": [t.tile_id for t in tiles if t.finetuned_split == "val"],
        "finetuned_train": [t.tile_id for t in tiles if t.finetuned_split == "train"],
    }
    reuse_summary = None
    if args.reuse_predictions:
        reuse_summary = {
            "dir": _rel(Path(args.reuse_predictions)),
            "allow_unverified": bool(args.allow_unverified_reuse),
            "reused": [t.tile_id for t in tiles if t.reused_prediction],
            "reused_unverified": [t.tile_id for t in tiles
                                  if t.reused_prediction
                                  and t.prediction_provenance == "legacy_unverified"],
            "regenerated": {t.tile_id: t.reuse_note for t in tiles
                            if not t.reused_prediction and t.reuse_note},
        }
    summary = {
        "protocol": args.protocol,
        "run": {
            "run_dir": _rel(run_dir),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "git": git,
            "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(elapsed, 1),
            "model_load_seconds": model_load_seconds,
            "seed": args.seed,
            "root": _rel(root),
            "calibration_pixel_cap": args.calibration_pixel_cap,
            "reuse_predictions": args.reuse_predictions,
            "reuse": reuse_summary,
            "save_predictions": bool(args.save_predictions),
            "save_comparisons": bool(args.save_comparisons),
            "reference": _rel(Path(args.reference)) if args.reference else None,
        },
        "backend": model_info,
        "tiles": {"total": len(tiles), "succeeded": len(succeeded), "errors": errors},
        "flags": flags,
        "overlap": overlap,
    }
    if args.protocol == "historical":
        summary["aggregates"] = protocol_out["aggregates"]
        summary["overall"] = protocol_out["overall"]
        if args.reference:
            summary["reference_comparison"] = compare_reference(succeeded, Path(args.reference),
                                                                args.compare_tol)
            _write_csv(run_dir / "comparison_vs_reference.csv",
                       summary["reference_comparison"]["per_tile"])
    else:
        if args.protocol == "geographic":
            summary["calibration"] = protocol_out["calibration"]
        else:
            summary["regions"] = protocol_out["regions"]
        summary["groups"] = groups
    summary["environment"] = _environment_info(args.seed)
    summary["limitations"] = _limitations(summary)

    manifest_rows = []
    for t in tiles:
        row = inventory_row(t)
        if args.protocol != "historical":
            row["protocol_b_split"] = t.protocol_b_split
        manifest_rows.append(row)
    _write_csv(run_dir / "manifest.csv", manifest_rows)
    _write_csv(run_dir / "per_tile.csv", rows)
    _write_csv(run_dir / "per_region.csv", [{"group": g, **a} for g, a in groups.items()])
    (run_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(summary["run"], indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(summary["environment"], indent=2),
                                              encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(_render_summary_md(summary, run_dir), encoding="utf-8")

    print(f"scored {len(succeeded)}/{len(tiles)} tiles in {elapsed:.0f}s; "
          f"errors={len(errors)}; summary: {run_dir / 'summary.md'}")
    return 0 if succeeded else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inventory", help="scan tile pairs, write the manifest template")
    pi.add_argument("--root", default=str(DEFAULT_ROOT))
    pi.add_argument("--out", default=None, help=f"CSV out (default {MANIFEST_TEMPLATE})")
    pi.add_argument("--ckpt", default=None, help="finetuned checkpoint for val-region labels")
    pi.add_argument("--stratum", action="append", help="only these strata (repeatable)")
    pi.set_defaults(func=cmd_inventory)

    pr = sub.add_parser("run", help="run a benchmark protocol")
    pr.add_argument("--protocol", required=True,
                    choices=["historical", "geographic", "local_calibration"])
    pr.add_argument("--backend", default="auto",
                    choices=["auto", "pix2pix", "imele", "finetuned", "depth_anything"]
                    + list(_CONTROL_BACKENDS) + list(_da3_map()))
    pr.add_argument("--model-path", default=None, help="override model/checkpoint path")
    pr.add_argument("--root", default=str(DEFAULT_ROOT))
    pr.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    pr.add_argument("--manifest", default=None,
                    help="CSV with tile_id,region,protocol_b_split overrides (e.g. the template)")
    pr.add_argument("--tiles", default=None, help="comma-separated tile ids to evaluate")
    pr.add_argument("--stratum", action="append", help="only these strata (repeatable)")
    pr.add_argument("--limit", type=int, default=None, help="first N tiles after sorting")
    pr.add_argument("--ckpt", default=None, help="finetuned checkpoint for overlap flags")
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--calibration-pixel-cap", type=int, default=2_000_000,
                    help="max pooled pixels for one calibration fit")
    pr.add_argument("--local-buffer-deg", type=float, default=0.2,
                    help="local_calibration: min bbox gap between calib and test tiles")
    pr.add_argument("--save-predictions", dest="save_predictions", action="store_true",
                    default=True, help="write predictions/<tile>_relative.npy (default on)")
    pr.add_argument("--no-predictions", dest="save_predictions", action="store_false")
    pr.add_argument("--save-comparisons", dest="save_comparisons", action="store_true",
                    default=True, help="write comparisons/<tile>.png (default on)")
    pr.add_argument("--no-comparisons", dest="save_comparisons", action="store_false")
    pr.add_argument("--reference", default=None,
                    help="historical: a scored_training_full.txt to compare against")
    pr.add_argument("--reuse-predictions", default=None,
                    help="directory of <tile>_relative.npy predictions; a cache is reused only "
                         "when its identity sidecar matches the tile inputs, model, seed and "
                         "inference-code fingerprint (otherwise the tile is re-inferred)")
    pr.add_argument("--allow-unverified-reuse", action="store_true",
                    help="accept cached predictions that have NO identity sidecar (legacy .npy "
                         "caches); recorded as legacy_unverified in the run metadata")
    pr.add_argument("--compare-tol", type=float, default=1e-3)
    pr.set_defaults(func=cmd_run)

    args = ap.parse_args()
    if getattr(args, "tiles", None):
        args.tiles = [t.strip() for t in args.tiles.split(",") if t.strip()]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
