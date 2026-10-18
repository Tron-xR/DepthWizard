"""Prediction generation for the Phase 3 HR test-set evaluation.

Reads the Phase 2 manifest (`server/data/manifest_hr.csv`) from the MAIN
checkout, read-only, and emits:

  runs/height_manifest_test.csv          - evaluation input for
                                           evaluation.py --mode ndsm|dsm
                                           (test split only; temporal-flagged
                                           tiles excluded like hr_headroom.md)
  runs/<backend>_<utc>/{run.json, model_info.json, environment.json,
                        predictions/<tile>_{ndsm,relative}.npy + sidecars}

One backend per process (the depth model cache is global and the RS backends
are single-select). RS arms emit nDSM metres (`_ndsm.npy`); da2/da3 and the
sobel/brightness controls emit relative depth (`_relative.npy`). Absolute
paths into the main checkout keep the Phase 2 dataset strictly read-only.

    .venv-rs\\Scripts\\python.exe experiments\\rs_eval\\run_hr_predictions.py \\
        --main-manifest ..\\DepthWizard\\server\\data\\manifest_hr.csv \\
        --backend rs3dada --device cuda --limit 1    # wiring check
    ... --backend sobel                              # controls need no model
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "server"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

BACKENDS = {  # CLI slug -> benchmark_tiles backend (None = control arm)
    "rs3dada": "rs3dada",
    "altimap_v1": "altimap_v1",
    "da2": "depth_anything",
    "da3": "da3-small",
    "sobel": "sobel",
    "brightness": "brightness",
}
_CONTROLS = ("sobel", "brightness")
_MANIFEST_COLUMNS = ("tile_id", "site_id", "stratum", "split",
                     "ndsm_path", "dsm_path", "prior_path", "rgb_path", "notes")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_height_manifest(main_manifest: Path, out_path: Path) -> tuple:
    """Project the Phase 2 manifest onto the evaluation columns.

    Paths become absolute references INTO THE MAIN CHECKOUT (read-only).
    Returns (rows, excluded_temporal_ids).
    """
    root = main_manifest.resolve().parents[2]
    rows, excluded = [], []
    with main_manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != "test":
                continue
            if (row.get("temporal_flag") or "False").strip() == "True":
                excluded.append(row["tile_id"])
                continue
            rows.append({
                "tile_id": row["tile_id"],
                "site_id": row["site_id"],
                "stratum": row["stratum"],
                "split": row["split"],
                "ndsm_path": str((root / row["ndsm_path"]).resolve()),
                "dsm_path": str((root / row["dsm_path"]).resolve()),
                "prior_path": str((root / row["coarse_dem_path"]).resolve()),
                "rgb_path": str((root / row["rgb_path"]).resolve()),
                "notes": "",
            })
    rows.sort(key=lambda r: (r["stratum"], r["tile_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-manifest", type=Path, required=True,
                    help="main checkout's server/data/manifest_hr.csv")
    ap.add_argument("--backend", choices=sorted(BACKENDS))
    ap.add_argument("--only-manifest", action="store_true",
                    help="build runs/height_manifest_test.csv and stop")
    ap.add_argument("--limit", type=int, default=0, help="first N tiles only")
    ap.add_argument("--device", default=None,
                    help="DEPTHWIZARD_RS_DEVICE for RS arms (default: auto)")
    ap.add_argument("--cuda-mem-fraction", default="0.85",
                    help="DEPTHWIZARD_CUDA_MEM_FRACTION for RS arms")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default=str(REPO_ROOT / "experiments"
                                              / "rs_eval" / "runs"))
    args = ap.parse_args()
    if not args.backend and not args.only_manifest:
        ap.error("pass --backend or --only-manifest")

    out_root = Path(args.out_root)
    manifest_path = out_root / "height_manifest_test.csv"
    rows, excluded = build_height_manifest(args.main_manifest, manifest_path)
    print(f"height manifest: {len(rows)} tiles -> {manifest_path} "
          f"(temporal-flagged excluded: {excluded})", flush=True)
    if args.only_manifest:
        return 0
    if args.limit:
        rows = rows[:args.limit]

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf-cache"))
    os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".torch-cache"))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if args.device:
        os.environ["DEPTHWIZARD_RS_DEVICE"] = args.device
    os.environ.setdefault("DEPTHWIZARD_CUDA_MEM_FRACTION", args.cuda_mem_fraction)

    import torch

    torch.set_num_threads(args.threads)

    import benchmark_tiles as bt  # noqa: E402  (tools/)

    backend = args.backend
    bt._force_backend(BACKENDS[backend], None)

    infer_fn = None
    if backend in _CONTROLS:
        infer_fn = bt._control_infer(backend)
    else:
        from app.pipeline import depth

        model = depth.ensure_model()
        slug = depth.backend_slug()
        if slug != BACKENDS[backend]:
            raise SystemExit(f"dispatch mismatch: {slug} != {BACKENDS[backend]}")
        infer_fn = bt._default_infer
        print(f"model loaded: {slug} ({depth.model_identifier()})", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{backend}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    identity = {"backend": backend, "seed": args.seed,
                "inference_fingerprint": bt._inference_fingerprint(),
                "model": bt._model_info(BACKENDS[backend])}
    (run_dir / "model_info.json").write_text(
        json.dumps(identity["model"], indent=2, default=str), encoding="utf-8")

    failed, per_tile_s = [], {}
    t_run = time.time()
    for i, row in enumerate(rows, 1):
        tile_id = row["tile_id"]
        rgb_path = Path(row["rgb_path"])
        ref_path = Path(row["ndsm_path"])
        tile = bt.Tile(tile_id, row["stratum"], rgb_path, ref_path)
        tile.info = {"rgb_sha256": _sha256(rgb_path),
                     "dem_sha256": _sha256(ref_path)}
        try:
            bt._apply_seed(args.seed)
            rgb = bt._load_rgb(rgb_path)
            t0 = time.time()
            pred = infer_fn(rgb)
            per_tile_s[tile_id] = round(time.time() - t0, 2)
            bt._write_prediction(run_dir, tile, pred, identity)
            print(f"[{i}/{len(rows)}] {tile_id} {pred.shape} "
                  f"{per_tile_s[tile_id]}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - count and continue
            failed.append({"tile_id": tile_id,
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i}/{len(rows)}] {tile_id} FAILED: {exc}", flush=True)

    (run_dir / "run.json").write_text(json.dumps({
        "backend": backend,
        "protocol": "hr-test-set predictions for evaluation.py --mode ndsm|dsm",
        "height_manifest": str(manifest_path),
        "height_manifest_sha256": _sha256(manifest_path),
        "tiles_requested": [r["tile_id"] for r in rows],
        "failed_tiles": failed,
        "per_tile_seconds": per_tile_s,
        "run_seconds": round(time.time() - t_run, 1),
        "seed": args.seed,
        "inference_fingerprint": identity["inference_fingerprint"],
        "git": bt._git_info(),
        "created_utc": stamp,
    }, indent=2, default=str), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(bt._environment_info(args.seed), indent=2, default=str),
        encoding="utf-8")
    print(f"run dir: {run_dir} ({len(per_tile_s)} ok, {len(failed)} failed)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
