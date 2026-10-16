#!/usr/bin/env python3
"""One-shot calibration sign-flip diagnostics (read-only evidence generator).

Investigates the seven pix2pix "calibration sign flip" tiles from the historical
benchmark comparison (benchmark_runs/c3a2841_pix2pix_20261016T134335Z) against
server/scored_training_full.txt (produced 2026-09-24 through the live /evaluate
route).

Method: replay every tile's cached float32 prediction through THREE fit
implementations on identical inputs:

  legacy-92b74a7  fit_affine as of 92b74a7 (Initial commit, 2026-09-10):
                  RANSAC inliers -> OLS refit, final scale taken verbatim.
  legacy-a28fedf  fit_affine as of a28fedf (2026-09-23): same fit PLUS the
                  low-relief degenerate guard. Still NO RANSAC-vs-OLS sign
                  reconciliation. (Diagnosis result: the reference does NOT
                  match this version on sparse_17, proving the scoring server
                  ran pre-a28fedf code; 92b74a7 is the reproducing baseline.)
  current         app.pipeline.calibration.fit_affine at HEAD: adds the
                  full-train OLS sign reconciliation (44295b8) and warnings.

If the legacy-a28fedf replay reproduces the historical calibrated values for
all 54 tiles (within TF-version float drift) while `current` differs only on
the flagged tiles, the flips are a fitting-behavior change, not nondeterminism.

Never writes into benchmark_runs/* except its own uniquely named
c3a2841_calibration_diagnostics_<UTC>/ output directory.

Run:
  cd server
  uv run python ../tools/diagnose_calibration.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import benchmark_tiles as bt  # noqa: E402

REPO_ROOT = bt.REPO_ROOT
DEFAULT_REFERENCE = REPO_ROOT / "server" / "scored_training_full.txt"
DEFAULT_PRED_DIR = (REPO_ROOT / "benchmark_runs" / "c3a2841_pix2pix_20261016T134335Z"
                    / "predictions")

_SPLIT_SEED = 42
_TRAIN_FRACTION = 0.8
_LOW_RELIEF_FRACTION = 0.02


# ---------------------------------------------------------------------------
# legacy fit emulations (see module docstring for commit provenance)
# ---------------------------------------------------------------------------
def _split(d: np.ndarray, h: np.ndarray) -> tuple:
    rng = np.random.default_rng(_SPLIT_SEED)
    order = rng.permutation(d.size)
    n_train = int(round(d.size * _TRAIN_FRACTION))
    return order[:n_train], order[n_train:]


def _legacy_fit(relative_d: np.ndarray, reference_h: np.ndarray, *,
                low_relief_guard: bool) -> dict:
    """Replay of pre-44295b8 fit_affine behavior on identical inputs."""
    d = np.asarray(relative_d).ravel().astype("float64")
    h = np.asarray(reference_h).ravel().astype("float64")
    mask = np.isfinite(d) & np.isfinite(h)
    d, h = d[mask], h[mask]
    if d.size < 16:
        raise ValueError("Not enough valid sample pairs for calibration")
    train, held = _split(d, h)
    d_train, h_train = d[train], h[train]
    X = d_train.reshape(-1, 1)

    def _degenerate(scale_zero_offset, method, n_inliers):
        return {"scale": 0.0, "offset": float(scale_zero_offset), "n_inliers": n_inliers,
                "degenerate": True, "method": method,
                "held_relative": d[held], "held_reference": h[held],
                "ransac_stats": None}

    if np.ptp(d_train) == 0.0:
        return _degenerate(np.median(h_train), "degenerate_constant", 0)
    ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=8,
                             residual_threshold=None, random_state=0)
    try:
        ransac.fit(X, h_train)
    except ValueError:
        return _degenerate(np.mean(h_train), "degenerate_constant", 0)
    inlier_mask = ransac.inlier_mask_
    lr = LinearRegression().fit(X[inlier_mask], h_train[inlier_mask])
    scale, offset = float(lr.coef_[0]), float(lr.intercept_)
    n_inliers = int(inlier_mask.sum())
    ransac_stats = _ransac_stats(X, h_train, inlier_mask)
    if (low_relief_guard and float(np.ptp(h_train)) > 0.0
            and abs(scale) * float(np.ptp(d_train)) < _LOW_RELIEF_FRACTION * float(np.ptp(h_train))):
        out = _degenerate(np.median(h_train), "degenerate_low_relief", n_inliers)
        out["ransac_stats"] = ransac_stats
        out["pre_guard_scale"] = scale
        return out
    return {"scale": scale, "offset": offset, "n_inliers": n_inliers,
            "degenerate": False, "method": "ransac_affine",
            "held_relative": d[held], "held_reference": h[held],
            "ransac_stats": ransac_stats}


def _ransac_stats(X: np.ndarray, h_train: np.ndarray, inlier_mask: np.ndarray) -> dict:
    inlier_fit = LinearRegression().fit(X[inlier_mask], h_train[inlier_mask])
    resid = h_train - inlier_fit.predict(X)
    ols_all = LinearRegression().fit(X, h_train)
    y_mad = float(np.median(np.abs(h_train - np.median(h_train))))
    return {
        "n_train": int(h_train.size),
        "n_inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "y_train_median_abs_dev": y_mad,
        "residual_abs_max_inliers": float(np.abs(resid[inlier_mask]).max()),
        "residual_abs_min_outliers": (float(np.abs(resid[~inlier_mask]).min())
                                      if (~inlier_mask).any() else None),
        "residual_abs_median": float(np.median(np.abs(resid))),
        "full_train_ols_scale": float(ols_all.coef_[0]),
        "full_train_ols_offset": float(ols_all.intercept_),
    }


def _fit_cal_fields(fit: dict, d_all: np.ndarray, h_all: np.ndarray) -> dict:
    """Mirror the /evaluate calibrated-mode fields for an emulated fit:
    degenerate -> raw all-pixel correlation, null metrics; else held-out only."""
    if fit["degenerate"]:
        corr = _corr(d_all, h_all)
        return {"cal_corr": corr, "rmse": None, "mae": None, "held": None}
    pred = fit["scale"] * fit["held_relative"] + fit["offset"]
    tru = fit["held_reference"]
    mae = float(np.mean(np.abs(pred - tru)))
    rmse = float(np.sqrt(np.mean((pred - tru) ** 2)))
    return {"cal_corr": _corr(pred, tru), "rmse": rmse, "mae": mae,
            "held": int(pred.size)}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _sign(v):
    if v is None:
        return None
    return 1 if v >= 0 else -1


# ---------------------------------------------------------------------------
# per-tile diagnosis
# ---------------------------------------------------------------------------
def diagnose_tile(tile: bt.Tile, pred_dir: Path) -> dict:
    from app.pipeline.calibration import fit_affine as fit_current
    from app.pipeline.evaluation import evaluate_prediction_truth

    npz = pred_dir / f"{tile.tile_id}_relative.npy"
    if not npz.is_file():
        return {"tile_id": tile.tile_id, "error": f"missing cached prediction: {npz}"}
    rdsm = np.load(npz)
    dem, dem_meta = bt._load_dem(tile.dem)
    rgb = bt._load_rgb(tile.rgb)
    dem = bt._align_pair(rgb, dem, dem_meta, bt._rgb_meta(tile.rgb))
    nodata = dem_meta["nodata"]

    row: dict = {
        "tile_id": tile.tile_id,
        "stratum": tile.stratum,
        "region": tile.region or "unassigned",
        "pred_sha256": bt.sha256_file(npz)[:16],
        "dem_sha256": tile.info["dem_sha256"][:16],
        "rgb_sha256": tile.info["rgb_sha256"][:16],
        "pred_shape": str(rdsm.shape),
        "dem_shape": str(dem.shape),
        "shapes_match": bool(rdsm.shape == dem.shape),
        "pred_dtype": str(rdsm.dtype),
        "pred_min": float(rdsm.min()), "pred_max": float(rdsm.max()),
        "pred_mean": float(rdsm.mean()), "pred_std": float(rdsm.std()),
        "nodata": nodata,
        "finite_pixels": int(np.isfinite(rdsm).sum()),
        "total_pixels": int(rdsm.size),
    }
    valid = np.isfinite(rdsm) & np.isfinite(dem)
    if nodata is not None and np.isfinite(nodata):
        valid &= dem != nodata
    d_all = rdsm[valid].astype("float64")
    h_all = dem[valid].astype("float64")
    row["valid_after_mask"] = int(d_all.size)
    row["raw_corr_all"] = _corr(d_all, h_all)

    order = _split(d_all, h_all)
    train, held = order
    row["raw_corr_train"] = _corr(d_all[train], h_all[train])
    row["raw_corr_held"] = _corr(d_all[held], h_all[held])

    old1 = _legacy_fit(rdsm, dem, low_relief_guard=False)
    old2 = _legacy_fit(rdsm, dem, low_relief_guard=True)
    new = fit_current(np.where(valid, rdsm, np.nan), np.where(valid, dem, np.nan))
    new_eval = evaluate_prediction_truth(rdsm, dem, nodata=nodata, mode="calibrated")

    old1_cal = _fit_cal_fields(old1, d_all, h_all)
    old2_cal = _fit_cal_fields(old2, d_all, h_all)

    rstats = old2["ransac_stats"] or old1["ransac_stats"]
    if rstats:
        row.update({
            "ransac_full_ols_scale": rstats["full_train_ols_scale"],
            "ransac_full_ols_offset": rstats["full_train_ols_offset"],
            "ransac_inlier_ratio": rstats["inlier_ratio"],
            "ransac_n_inliers": rstats["n_inliers"],
            "ransac_n_train": rstats["n_train"],
            "ransac_y_mad": rstats["y_train_median_abs_dev"],
            "ransac_inlier_absres_max": rstats["residual_abs_max_inliers"],
            "ransac_outlier_absres_min": rstats["residual_abs_min_outliers"],
            "ransac_resid_abs_median": rstats["residual_abs_median"],
            "ransac_sign_vs_ols": _sign(rstats["full_train_ols_scale"]),
        })
    row.update({
        "old1_scale": old1["scale"], "old1_degenerate": old1["degenerate"],
        "old1_method": old1["method"], "old1_n_inliers": old1["n_inliers"],
        "old1_cal_corr": old1_cal["cal_corr"], "old1_rmse": old1_cal["rmse"],
        "old1_mae": old1_cal["mae"],
        "old2_scale": old2["scale"], "old2_degenerate": old2["degenerate"],
        "old2_method": old2["method"], "old2_n_inliers": old2["n_inliers"],
        "old2_cal_corr": old2_cal["cal_corr"], "old2_rmse": old2_cal["rmse"],
        "old2_mae": old2_cal["mae"],
        "new_scale": new.scale, "new_offset": new.offset,
        "new_status": new.calibration_status, "new_warning": new.calibration_warning,
        "new_degenerate": new.degenerate, "new_method": new.method,
        "new_cal_corr": new_eval["correlation"], "new_rmse": new_eval["rmse"],
        "new_mae": new_eval["mae"],
        "new_raw_corr_signed": new_eval["raw_correlation_signed"],
    })
    return row


def _classify(row: dict) -> str:
    ref_cal = row.get("ref_cal_corr")
    new_cal = row.get("new_cal_corr")
    old2_cal = row.get("old2_cal_corr")
    if ref_cal is None:
        if row.get("new_degenerate") and row.get("old1_cal_corr") is None:
            # No historical calibrated correlation either (the scale-0 fit
            # produced a constant prediction, so the correlation was undefined
            # on BOTH sides) and the current fit is a flagged degenerate.
            # Both eras degenerate on this median-plateau tile: not a flip.
            return "degenerate_in_both_eras"
        return "undefined"
    if new_cal is None or old2_cal is None:
        return "undefined"
    if _sign(new_cal) == _sign(ref_cal):
        return "reproduced"
    if row["new_degenerate"]:
        return "changed_to_new_guard_degenerate"
    if _sign(old2_cal) == _sign(ref_cal):
        return "changed_by_sign_reconciliation"
    return "unexplained"


# ---------------------------------------------------------------------------
# extended diagnostics for the flip set
# ---------------------------------------------------------------------------
def extended_diagnostics(tile: bt.Tile, pred_dir: Path, trials: int) -> dict:
    from app.pipeline.calibration import fit_affine as fit_current

    rdsm = np.load(pred_dir / f"{tile.tile_id}_relative.npy").astype("float64")
    dem, dem_meta = bt._load_dem(tile.dem)
    rgb = bt._load_rgb(tile.rgb)
    dem = bt._align_pair(rgb, dem, dem_meta, bt._rgb_meta(tile.rgb))
    valid = np.isfinite(rdsm) & np.isfinite(dem)

    out: dict = {"tile_id": tile.tile_id}

    # 1) repeated identical fits -> determinism
    scales_new = [fit_current(rdsm, dem).scale for _ in range(3)]
    scales_old = [_legacy_fit(rdsm, dem, low_relief_guard=True)["scale"] for _ in range(3)]
    out["repeat_new_identical"] = bool(np.all(np.asarray(scales_new) == scales_new[0]))
    out["repeat_old_identical"] = bool(np.all(np.asarray(scales_old) == scales_old[0]))
    out["repeat_new_scales"] = scales_new

    # 2) float32 vs float64 (cache dtype vs upcast; same VALUES -> must agree)
    fit32 = fit_current(rdsm.astype("float32"), dem)
    fit64 = fit_current(rdsm, dem)
    out["float32_vs_float64_same"] = bool(fit32.scale == fit64.scale)

    # 3) order sensitivity: permute the flattened pixel order (changes which
    #    pixels land in the fixed-seed 80/20 split -> not expected to be
    #    invariant; recorded to bound it)
    order_flips = {"new": 0, "old": 0}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(rdsm.size)
        d_p = rdsm.ravel()[idx].reshape(rdsm.shape)
        h_p = dem.ravel()[idx].reshape(dem.shape)
        if fit_current(d_p, h_p).scale < 0:
            order_flips["new"] += 1
        if _legacy_fit(d_p, h_p, low_relief_guard=True)["scale"] < 0:
            order_flips["old"] += 1
    out["order_permutation_negative_scale_counts_of_3"] = order_flips

    # 4) TF-drift simulation: gaussian noise at realistic relative magnitudes
    ptp = float(np.ptp(rdsm))
    drift = {}
    for sigma in (1e-6, 1e-5, 1e-4):
        counts = {"new": 0, "old": 0}
        for seed in range(trials):
            rng = np.random.default_rng(1000 + seed)
            rdsm_p = rdsm + rng.normal(0.0, sigma * ptp, rdsm.shape)
            if fit_current(rdsm_p, dem).scale < 0:
                counts["new"] += 1
            if _legacy_fit(rdsm_p, dem, low_relief_guard=True)["scale"] < 0:
                counts["old"] += 1
        drift[f"sigma_{sigma:g}"] = {**counts, "trials": trials}
    out["drift_negative_scale_counts"] = drift
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(bt.DEFAULT_ROOT))
    ap.add_argument("--predictions", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--run-root", default=str(bt.DEFAULT_RUN_ROOT))
    ap.add_argument("--trials", type=int, default=5,
                    help="perturbation trials per sigma for the flip tiles")
    ap.add_argument("--skip-extended", action="store_true")
    args = ap.parse_args()

    pred_dir = Path(args.predictions)
    if not pred_dir.is_dir():
        raise SystemExit(f"predictions dir not found: {pred_dir}")
    ref_path = Path(args.reference)
    if not ref_path.is_file():
        raise SystemExit(f"reference not found: {ref_path}")

    started = datetime.now(timezone.utc)
    git = bt._git_info()
    run_dir = Path(args.run_root) / (
        f"{git['commit'][:7]}_calibration_diagnostics_{started.strftime('%Y%m%dT%H%M%SZ')}")
    run_dir.mkdir(parents=True)

    ref_rows = {r["tile_id"]: r for r in bt._parse_reference(ref_path)}
    tiles, _ = bt.scan_tiles(Path(args.root))
    print(f"tiles: {len(tiles)}; predictions: {pred_dir}", flush=True)

    rows = []
    for tile in tiles:
        bt.inspect_tile(tile)
        row = diagnose_tile(tile, pred_dir)
        r = ref_rows.get(tile.tile_id, {})
        row["ref_raw_corr"] = r.get("raw_corr")
        row["ref_cal_corr"] = r.get("cal_corr")
        row["ref_rmse"] = r.get("rmse")
        row["ref_mae"] = r.get("mae")
        row["ref_flag"] = r.get("flag")
        row["classification"] = _classify(row) if "error" not in row else row.get("error")
        rows.append(row)
        print(f"  {tile.tile_id}: ref={row.get('ref_cal_corr')} old2={row.get('old2_cal_corr')} "
              f"new={row.get('new_cal_corr')} -> {row.get('classification')}", flush=True)

    flips = [r for r in rows if r.get("classification") in
             ("changed_by_sign_reconciliation", "changed_to_new_guard_degenerate",
              "unexplained")]
    flips.sort(key=lambda r: r["tile_id"])

    detail = {"flip_tiles": [r["tile_id"] for r in flips], "tiles": {}}
    if not args.skip_extended:
        for tile in tiles:
            if tile.tile_id in detail["flip_tiles"]:
                detail["tiles"][tile.tile_id] = extended_diagnostics(
                    tile, pred_dir, args.trials)
                print(f"  extended: {tile.tile_id} done", flush=True)

    bt._write_csv(run_dir / "diagnostics_all.csv", rows)
    (run_dir / "flips_detail.json").write_text(json.dumps(detail, indent=2, default=str),
                                               encoding="utf-8")

    matched = [r for r in rows if r.get("old2_cal_corr") is not None
               and r.get("ref_cal_corr") is not None]
    old2_deltas = [abs(r["old2_cal_corr"] - r["ref_cal_corr"]) for r in matched]
    new_deltas = [abs(r["new_cal_corr"] - r["ref_cal_corr"]) for r in matched
                  if r.get("new_cal_corr") is not None]
    old2_sign_match = sum(1 for r in matched
                          if _sign(r["old2_cal_corr"]) == _sign(r["ref_cal_corr"]))
    new_sign_match = sum(1 for r in matched
                         if _sign(r["new_cal_corr"]) == _sign(r["ref_cal_corr"]))
    matched1 = [r for r in rows if r.get("old1_cal_corr") is not None
                and r.get("ref_cal_corr") is not None]
    old1_deltas = [abs(r["old1_cal_corr"] - r["ref_cal_corr"]) for r in matched1]
    old1_sign_match = sum(1 for r in matched1
                          if _sign(r["old1_cal_corr"]) == _sign(r["ref_cal_corr"]))
    classes: dict = {}
    for r in rows:
        classes[r.get("classification")] = classes.get(r.get("classification"), 0) + 1

    summary = {
        "run_dir": bt._rel(run_dir),
        "git": git,
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tiles": len(rows),
        "reference": bt._rel(ref_path),
        "predictions_dir": bt._rel(pred_dir),
        "flip_tiles": detail["flip_tiles"],
        "classification_counts": classes,
        "legacy_92b74a7_replay": {
            "tiles_compared": len(matched1),
            "sign_matches_vs_reference": old1_sign_match,
            "max_abs_delta_cal_corr": max(old1_deltas) if old1_deltas else None,
            "median_abs_delta_cal_corr": float(np.median(old1_deltas)) if old1_deltas else None,
        },
        "legacy_a28fedf_replay": {
            "tiles_compared": len(matched),
            "sign_matches_vs_reference": old2_sign_match,
            "max_abs_delta_cal_corr": max(old2_deltas) if old2_deltas else None,
            "median_abs_delta_cal_corr": float(np.median(old2_deltas)) if old2_deltas else None,
        },
        "current_fit_vs_reference": {
            "sign_matches_vs_reference": new_sign_match,
            "max_abs_delta_cal_corr": max(new_deltas) if new_deltas else None,
        },
        "extended": detail["tiles"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")

    lines = [
        "# Calibration sign-flip diagnostics",
        "",
        f"- commit: `{git['commit'][:12]}` (dirty={git['dirty']})",
        f"- predictions: `{summary['predictions_dir']}`",
        f"- reference: `{summary['reference']}`",
        f"- tiles diagnosed: {summary['tiles']}",
        f"- flip tiles (new sign != reference sign): {summary['flip_tiles']}",
        "",
        "## Verdict",
        "",
        f"- legacy-92b74a7 replay sign-vs-reference matches: "
        f"{old1_sign_match}/{len(matched1)}; max |delta cal_corr| = "
        f"{summary['legacy_92b74a7_replay']['max_abs_delta_cal_corr']}",
        f"- legacy-a28fedf replay sign-vs-reference matches: "
        f"{old2_sign_match}/{len(matched)}; max |delta cal_corr| = "
        f"{summary['legacy_a28fedf_replay']['max_abs_delta_cal_corr']}",
        f"- current fit sign-vs-reference matches: {new_sign_match}/{len(matched)}",
        f"- classification counts: {classes}",
        "",
        "| tile | ref cal | old2 cal | new cal | old2 scale | new scale "
        "| new status | classification |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["tile_id"]):
        lines.append(
            f"| {r['tile_id']} | {bt._fmt(r.get('ref_cal_corr'))} "
            f"| {bt._fmt(r.get('old2_cal_corr'))} | {bt._fmt(r.get('new_cal_corr'))} "
            f"| {bt._fmt(r.get('old2_scale'), 2)} | {bt._fmt(r.get('new_scale'), 2)} "
            f"| {r.get('new_status')} | {r.get('classification')} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"old1 (92b74a7, no guard) replay: {old1_sign_match}/{len(matched1)} sign matches, "
          f"max |delta|={summary['legacy_92b74a7_replay']['max_abs_delta_cal_corr']:.2e}")
    print(f"old2 replay: {old2_sign_match}/{len(matched)} sign matches, "
          f"max |delta|={summary['legacy_a28fedf_replay']['max_abs_delta_cal_corr']:.2e}")
    print(f"new fit: {new_sign_match}/{len(matched)} sign matches")
    print(f"classes: {classes}")
    print(f"diagnostics: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
