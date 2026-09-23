"""Optional prediction-artifact export for diagnosing the model output itself.

When a request opts in (e.g. ``/evaluate?save_artifacts=true``), the pipeline
writes the EXACT numerical prediction the evaluator/renderer consumed, before
any 3D-mesh / heightmap conversion, plus a visualization and a metrics dump.

Files written under ``<OUTPUT_DIR>/<backend>/<input-stem>/``:

  predicted_depth.npy   - the FINAL prediction (calibrated absolute elevation
                          when one exists, else the raw relative depth). Raw
                          float values, exact dtype preserved, never uint8,
                          never re-normalized. This is the canonical artifact.
  raw_prediction.npy    - stage A: model output on the native grid, before
                          inversion and before 0..1 normalization.
  relative_prediction.npy - stage B: normalized 0..1 relative depth/rDSM.
  calibrated_prediction.npy - stage C: calibrated absolute metric prediction
                          (only when calibration ran).
  predicted_depth.png   - VISUALIZATION ONLY (robust percentile stretch on a
                          copy). Never used for metrics.
  predicted_depth.tif   - single-band float georeferenced prediction sharing
                          the input's CRS/transform/bounds; a plain
                          non-georeferenced TIFF when the input carried none.
  metrics.json          - input/backend/model info, prediction stats, the
                          reported evaluation metrics, and calibration health.

Stage arrays that are not available for the caller (e.g. /validate only has the
persisted absolute DSM) are simply not written; nothing is re-inferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import exporter

_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _stats(arr: np.ndarray, percentiles: tuple = _PERCENTILES) -> dict:
    """A JSON-safe stats block: shape/dtype + finite min/max/mean/std/p*/unique."""
    a = np.asarray(arr)
    finite = a[np.isfinite(a)].astype("float64")
    out = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if finite.size == 0:
        out.update({"min": None, "max": None, "mean": None, "std": None,
                    "unique": 0})
    else:
        out.update({
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "unique": int(np.unique(finite).size),
        })
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(finite, p)) if finite.size else None
    return out


def _robust_8bit(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0,
                 nan_fill: int = 128) -> tuple:
    """Map a prediction to a 0..255 visualization COPY (never the prediction
    itself). Robust percentile stretch; NaN/inf become neutral grey.
    Returns (viz_uint8_2d, viz_min, viz_max, p_low, p_high)."""
    a = np.asarray(arr, dtype="float64")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.full(a.shape, nan_fill, dtype="uint8"), 0.0, 1.0, p_low, p_high
    lo = float(np.percentile(finite, p_low))
    hi = float(np.percentile(finite, p_high))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    viz = np.full(a.shape, nan_fill, dtype="uint8")
    good = np.isfinite(a)
    viz[good] = np.clip((a[good] - lo) / (hi - lo) * 255.0, 0, 255).astype("uint8")
    return viz, lo, hi, p_low, p_high


def export_prediction_artifacts(
    *,
    out_root: Path,
    backend: str,
    model_identifier: str,
    input_filename: str,
    job_id: Optional[str] = None,
    relative: Optional[np.ndarray] = None,
    calibrated: Optional[np.ndarray] = None,
    raw: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
    crs: Optional[str] = None,
    transform=None,
    bounds=None,
) -> str:
    """Write the prediction-artifact bundle for one input.

    ``final`` mirrors exactly what evaluation/renderer used: the calibrated
    absolute prediction when present, else the relative depth. Returns the
    output directory containing the artifacts.
    """
    stem = Path(input_filename).stem
    out_dir = out_root / backend / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    final = calibrated if calibrated is not None else relative
    if final is None:
        raise ValueError("Nothing to export: no calibrated or relative prediction given")

    stage_files = {}
    for name, arr in (("raw", raw), ("relative", relative), ("calibrated", calibrated)):
        if arr is not None:
            path = out_dir / f"{name}_prediction.npy"
            np.save(path, np.asarray(arr))
            stage_files[name] = path

    predicted_npy = out_dir / "predicted_depth.npy"
    np.save(predicted_npy, np.asarray(final))

    # ---- visualization (copy only; never used for metrics) ----------------- #
    viz, viz_min, viz_max, p_low, p_high = _robust_8bit(final)
    from PIL import Image

    predicted_png = out_dir / "predicted_depth.png"
    Image.fromarray(viz, "L").save(predicted_png, "PNG")

    # ---- georeferenced single-band float TIFF (same array/extent as input) -#
    predicted_tif = out_dir / "predicted_depth.tif"
    exporter.export_geotiff(np.asarray(final).astype("float32"), crs, transform,
                            bounds, predicted_tif)

    # ---- metrics.json ------------------------------------------------------ #
    resolution = None
    if transform is not None:
        try:
            resolution = [abs(float(transform.a)), abs(float(transform.e))]
        except Exception:
            resolution = None
    payload = {
        "input_filename": input_filename,
        "backend": backend,
        "model": model_identifier,
        "georeferenced": bool(crs and transform is not None),
        "crs": crs,
        "resolution": resolution,
        "input_dimensions": list(np.asarray(final).shape),
        "prediction_dimensions": list(np.asarray(final).shape),
        "prediction": _stats(final),
        "stages": {name: _stats(arr) for name, arr in
                   (("raw", raw), ("relative", relative), ("calibrated", calibrated))
                   if arr is not None},
        "ground_truth": _stats(ground_truth) if ground_truth is not None else None,
        "visualization": {
            "method": "robust percentile stretch (copy only)",
            "percentiles": [p_low, p_high],
            "min": viz_min,
            "max": viz_max,
        },
    }
    if metrics:
        # Keep exactly the reported evaluation metric fields; evaluation already
        # separates calibration health, so a nested copy is transparent.
        payload["metrics"] = dict(metrics)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary(payload, out_dir, stage_files)
    return str(out_dir)


def _print_summary(payload: dict, out_dir: Path, stage_files: dict) -> None:
    pred = payload["prediction"]
    print("=" * 60)
    print("PREDICTION ARTIFACT EXPORT")
    print("-" * 60)
    print(f"input      : {payload['input_filename']}")
    print(f"backend    : {payload['backend']}  ({payload['model']})")
    print(f"prediction : shape={pred['shape']} dtype={pred['dtype']}")
    print(f"  min/max  : {pred['min']} / {pred['max']}")
    print(f"  mean/std : {pred['mean']} / {pred['std']}")
    print(f"  percentiles p1/p5/p25/p50/p75/p95/p99: "
          + " ".join(f"{pred[f'p{p}']:.4g}" for p in _PERCENTILES))
    print(f"  unique   : {pred['unique']}")
    viz = payload["visualization"]
    print(f"visualiz. : p{viz['percentiles'][0]:g}..p{viz['percentiles'][1]:g} "
          f"stretch -> 0..255, viz_min={viz['min']:.4g} viz_max={viz['max']:.4g} "
          "(copy only, NOT used in metrics)")
    if payload.get("ground_truth"):
        gt = payload["ground_truth"]
        print(f"gt truth  : shape={gt['shape']} min/max/mean/std "
              f"= {gt['min']} / {gt['max']} / {gt['mean']} / {gt['std']}")
    georef = "yes" if payload["georeferenced"] else "NO (plain TIFF)"
    print(f"georeferenced output: {georef}  crs={payload['crs']}")
    print(f"stages    : {', '.join(f'{k}_prediction.npy' for k in stage_files) or 'final-only'}")
    print(f"artifacts : {out_dir}")
    print("=" * 60)