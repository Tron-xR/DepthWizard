"""Headroom analysis on the HR dataset test split (Phase 2).

Everything is scored against each tile's ``dsm_hr`` - the reference path
recorded in the manifest as ``reference_hr_path`` - on the built valid mask,
split into spatial calibration/evaluation blocks so every oracle correction is
fitted without touching the evaluation pixels. Two families, reported
separately:

  priors    real downloaded products (GLO-30 coarse, 90/250 m national DEMs)
            with their raw, datum-offset-corrected, zero-mean-oracle and
            affine-oracle arms, a constant arm, a perfect-terrain floor from
            the built ndsm (dsm = dtm + ndsm by construction), and two
            composition oracles: prior_90m + true ndsm (the 90 m prior's
            terrain error with a perfect depth model on top) and
            prior_90m + true residual (an identity sanity check).
  backends  the four deployed/opt-in depth backends through the deployed
            calibration workflow (``evaluation.evaluate_tile_arrays``). The HR
            scenes sit far outside their training domain: these are a floor
            expectation for Phase 3, not a tuning target.

Temporal-flagged tiles stay listed but are excluded from the primary
aggregates. Outputs: JSON + CSV under server/data/, and with --report a
markdown summary at docs/benchmarks/hr_headroom.md.

CLI (from server/):
    uv run python -m app.pipeline.headroom [--skip-backends] [--report]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from app.pipeline import calibration as calib
from app.pipeline import dem_source, depth, evaluation, hr_dataset

REPO_ROOT = hr_dataset.REPO_ROOT
BLOCK = 180  # mirrors the acceptance evaluator's spatial split
GUARD = 8
PRIOR_COLUMNS = (
    ("coarse_dem", "coarse_dem_path"),
    ("prior_90m", "prior_90m_path"),
    ("prior_250m", "prior_250m_path"),
)
BACKENDS = ("pix2pix", "depth_anything", "finetuned", "imele")
DEFAULT_FINETUNED_CKPT = (REPO_ROOT / "decoder_training" / "checkpoints"
                          / "da2_decoder_finetuned.pt")
DEFAULT_IMELE_MODEL = REPO_ROOT / "models" / "imele_model.tar"


def _abs(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else REPO_ROOT / path


def _read(rel: str) -> np.ndarray:
    data, _meta = dem_source._read_local_geotiff(str(_abs(rel)))
    return data


def _read_with_meta(rel: str):
    return dem_source._read_local_geotiff(str(_abs(rel)))


def _read_rgb(rel: str) -> np.ndarray:
    import rasterio

    with rasterio.open(_abs(rel)) as src:
        bands = [src.read(i) for i in range(1, min(src.count, 3) + 1)]
    return np.dstack(bands)


def _arm(pred, truth, mask) -> dict:
    stats = evaluation._arm_stats(np.asarray(pred, dtype="float32"), truth, mask)
    return {"n": stats["n"], "rmse": stats["rmse"], "mae": stats["mae"],
            "bias": stats["bias"], "r": stats["correlation"]}


def tile_priors(row: dict) -> dict:
    """All prior/floor arms of one tile, fitted on calib blocks, scored on eval."""
    dem = _read(row["reference_hr_path"]).astype("float64")
    dtm = _read(row["dtm_path"]).astype("float64")
    ndsm = _read(row["ndsm_path"]).astype("float64")
    mask = _read(row["mask_path"])
    valid = (mask > 0) & np.isfinite(dem) & np.isfinite(dtm) & np.isfinite(ndsm)
    calib_mask, eval_mask = evaluation.spatial_block_split(
        dem.shape, block=BLOCK, guard=GUARD)
    calib_px = calib_mask & valid
    eval_px = eval_mask & valid

    record = {
        "tile_id": row["tile_id"], "site_id": row["site_id"],
        "split": row["split"], "stratum": row["stratum"],
        "temporal_flag": row["temporal_flag"],
        "valid_frac": float(row["valid_frac"]),
        "datum_offset_m": float(row["datum_offset_m"] or 0.0),
        "datum_offset_method": row["datum_offset_method"],
        "n_calib_pixels": int(calib_px.sum()), "n_eval_pixels": int(eval_px.sum()),
        "arms": {}, "decomposition": None, "status": "ok",
    }
    if int(eval_px.sum()) < 2 or int(calib_px.sum()) < 2:
        record["status"] = "no_eval_pixels"
        return record

    const = float(np.mean(dem[calib_px]))
    record["arms"]["const_mean"] = _arm(np.full(dem.shape, const), dem, eval_px)

    for name, column in PRIOR_COLUMNS:
        prior = _read(row[column]).astype("float64")
        fit_px = calib_px & np.isfinite(prior)
        score_px = eval_px & np.isfinite(prior)
        if int(fit_px.sum()) < 16 or int(score_px.sum()) < 2:
            record["arms"][name] = {"status": "insufficient_pixels"}
            continue
        record["arms"][name] = _arm(prior, dem, score_px)

        residual = prior - dem
        mu = float(np.mean(residual[fit_px]))
        record["arms"][name + "_zeromean"] = _arm(prior - mu, dem, score_px)
        try:
            fit = calib.fit_affine(np.where(fit_px, prior, np.nan),
                                   np.where(fit_px, dem, np.nan))
        except ValueError:
            fit = None
        if fit is not None and not fit.degenerate:
            record["arms"][name + "_affine"] = _arm(
                fit.scale * prior + fit.offset, dem, score_px)
        else:
            record["arms"][name + "_affine"] = {"status": "degenerate_fit"}

    coarse = _read(row["coarse_dem_path"]).astype("float64")
    offset = record["datum_offset_m"]
    score_px = eval_px & np.isfinite(coarse)
    record["arms"]["coarse_dem_offset_corrected"] = _arm(
        coarse + offset, dem, score_px)

    canopy_const = float(np.mean(ndsm[calib_px]))
    record["arms"]["perfect_terrain_floor"] = _arm(
        dtm + canopy_const, dem, eval_px)

    prior_90 = _read(row["prior_90m_path"]).astype("float64")
    residual = _read(row["residual_path"]).astype("float64")
    record["arms"]["prior_90m_plus_true_ndsm"] = _arm(
        prior_90 + ndsm, dem, eval_px)
    record["arms"]["prior_90m_plus_true_residual"] = _arm(
        prior_90 + residual, dem, eval_px)

    v_dsm = float(np.var(dem[eval_px]))
    v_dtm = float(np.var(dtm[eval_px]))
    v_ndsm = float(np.var(ndsm[eval_px]))
    a = dtm[eval_px] - float(np.mean(dtm[eval_px]))
    b = ndsm[eval_px] - float(np.mean(ndsm[eval_px]))
    cov = float(np.mean(a * b))
    record["decomposition"] = {
        "var_dsm": v_dsm, "var_dtm": v_dtm, "var_ndsm": v_ndsm,
        "cov_dtm_ndsm": cov, "std_ndsm": float(np.sqrt(v_ndsm)),
        "canopy_var_share": (v_ndsm + 2.0 * cov) / v_dsm if v_dsm else None,
    }
    return record


def _force_backend(name: str):
    """Point app.config at one backend; None on success, else the reason."""
    from app import config

    config.FINETUNED_MODEL_PATH = ""
    config.IMELE_MODEL_PATH = ""
    for _slug, attr, _default in depth.DA3_BACKENDS:
        setattr(config, attr, "")
    reason = None
    if name == "pix2pix":
        if not config.TF_MODEL_PATH:
            reason = "no pix2pix SavedModel discovered"
    elif name == "depth_anything":
        config.TF_MODEL_PATH = ""
    elif name == "finetuned":
        if not DEFAULT_FINETUNED_CKPT.exists():
            reason = f"missing checkpoint {DEFAULT_FINETUNED_CKPT}"
        else:
            config.FINETUNED_MODEL_PATH = str(DEFAULT_FINETUNED_CKPT)
            config.TF_MODEL_PATH = ""
    elif name == "imele":
        if not DEFAULT_IMELE_MODEL.exists():
            reason = f"missing model {DEFAULT_IMELE_MODEL}"
        else:
            config.IMELE_MODEL_PATH = str(DEFAULT_IMELE_MODEL)
            config.TF_MODEL_PATH = ""
    depth._model_cache.clear()
    return reason


def run_backends(rows: list, cache_dir) -> dict:
    results: dict = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in BACKENDS:
        reason = _force_backend(name)
        if reason:
            results[name] = {"availability": {"status": "unavailable",
                                              "reason": reason}, "tiles": {}}
            print(f"[backends] {name}: {reason}")
            continue
        results[name] = {"availability": {"status": "ok"}, "tiles": {}}
        for row in rows:
            tile_id = row["tile_id"]
            try:
                cached = cache_dir / f"{name}_{tile_id}.npy"
                if cached.exists():
                    rdsm = np.load(cached)
                    print(f"[backends] {name} {tile_id}: cached")
                else:
                    rdsm = depth.infer_relative_dsm(_read_rgb(row["rgb_path"]))
                    np.save(cached, rdsm)
                dem, meta = _read_with_meta(row["reference_hr_path"])
                georef = (meta["transform"], meta["crs"],
                          meta["transform"], meta["crs"])
                scored = evaluation.evaluate_tile_arrays(
                    dem, {name: rdsm}, georef=georef,
                    gsd_m=float(row["gsd"]))
                arm = dict(scored["backends"][name])
                arm["n_eval_pixels"] = scored["n_eval_pixels"]
                results[name]["tiles"][tile_id] = arm
                print(f"[backends] {name} {tile_id}: "
                      f"raw_r={arm.get('raw_r')} cal_rmse={arm.get('cal_rmse')}")
            except Exception as exc:
                results[name]["tiles"][tile_id] = {
                    "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                print(f"[backends] {name} {tile_id}: error {exc}")
    return results


def _agg(values: list) -> dict:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(vals), "mean": float(np.mean(vals)),
            "median": float(np.median(vals))}


def _prior_aggregates(tile_records: list) -> dict:
    primary = [t for t in tile_records if t["status"] == "ok"
               and str(t["temporal_flag"]) != "True"]
    aggregates = {"n_primary_tiles": len(primary),
                  "excluded_temporal": [t["tile_id"] for t in tile_records
                                        if str(t["temporal_flag"]) == "True"],
                  "overall": {}, "by_stratum": {}}
    arm_names = sorted({name for t in primary for name in t["arms"]})
    for name in arm_names:
        aggregates["overall"][name] = {
            "rmse": _agg([t["arms"][name].get("rmse") for t in primary]),
            "mae": _agg([t["arms"][name].get("mae") for t in primary]),
        }
    for stratum in sorted({t["stratum"] for t in primary}):
        group = [t for t in primary if t["stratum"] == stratum]
        aggregates["by_stratum"][stratum] = {
            "n_tiles": len(group),
            "arms": {name: {
                "rmse": _agg([t["arms"][name].get("rmse")
                              for t in group if name in t["arms"]]),
                "mae": _agg([t["arms"][name].get("mae")
                             for t in group if name in t["arms"]]),
            } for name in arm_names},
            "decomposition": {
                key: _agg([t["decomposition"][key] for t in group
                           if t["decomposition"]]) for key in
                ("std_ndsm", "canopy_var_share", "var_dsm", "var_dtm")},
        }
    return aggregates


def _backend_aggregates(results: dict, tile_records: list) -> dict:
    primary = {t["tile_id"]: t for t in tile_records if t["status"] == "ok"
               and str(t["temporal_flag"]) != "True"}
    aggregates = {}
    for name, payload in results.items():
        if payload["availability"]["status"] != "ok":
            aggregates[name] = payload["availability"]
            continue
        arms = {tid: arm for tid, arm in payload["tiles"].items()
                if tid in primary}
        aggregates[name] = {
            "status": "ok",
            "raw_r": _agg([a.get("raw_r") for a in arms.values()]),
            "raw_rmse": _agg([a.get("raw_rmse") for a in arms.values()]),
            "cal_rmse": _agg([a.get("cal_rmse") for a in arms.values()]),
            "cal_mae": _agg([a.get("cal_mae") for a in arms.values()]),
            "cal_r": _agg([a.get("cal_r") for a in arms.values()]),
            "by_stratum": {
                stratum: {
                    "raw_r": _agg([a.get("raw_r") for tid, a in arms.items()
                                   if primary[tid]["stratum"] == stratum]),
                    "cal_rmse": _agg([a.get("cal_rmse") for tid, a in arms.items()
                                      if primary[tid]["stratum"] == stratum]),
                } for stratum in sorted({t["stratum"] for t in primary.values()})},
        }
    return aggregates


def _fmt(value, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _render_md(summary: dict) -> str:
    lines = [
        "# HR dataset headroom (Phase 2)",
        "",
        f"Generated {summary['generated']} from `server/data/manifest_hr.csv`.",
        "All arms scored against `dsm_hr` (the manifest `reference_hr_path`) on",
        "each tile's built valid mask; oracle corrections are fitted on spatial",
        "calibration blocks, metrics on held-out evaluation blocks.",
        f"Primary set: {summary['priors']['n_primary_tiles']} test tiles; "
        f"temporal-flagged excluded: {summary['priors']['excluded_temporal']}.",
        "",
        "## Priors vs dsm_hr (RMSE m, overall median of per-tile values)",
        "",
        "| arm | median RMSE | median MAE |",
        "| --- | --- | --- |",
    ]
    for name, agg in summary["priors"]["overall"].items():
        lines.append(f"| {name} | {_fmt(agg['rmse']['median'], 2)} "
                     f"| {_fmt(agg['mae']['median'], 2)} |")
    lines += ["", "### Per stratum (median RMSE m)", "",
              "| arm | " + " | ".join(summary["priors"]["by_stratum"]) + " |",
              "| --- |" + " --- |" * len(summary["priors"]["by_stratum"])]
    arm_names = list(summary["priors"]["overall"])
    for name in arm_names:
        cells = [_fmt(summary["priors"]["by_stratum"][s]["arms"][name]
                      ["rmse"]["median"], 2)
                 for s in summary["priors"]["by_stratum"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "### Variance decomposition (median over primary tiles)", "",
              "| stratum | std(dsm) | std(dtm) | std(ndsm) | canopy share of var |",
              "| --- | --- | --- | --- | --- |"]
    for stratum, agg in summary["priors"]["by_stratum"].items():
        decomp = agg["decomposition"]
        lines.append(
            f"| {stratum} | {_fmt(decomp['var_dsm']['median'] ** 0.5, 1)} "
            f"| {_fmt(decomp['var_dtm']['median'] ** 0.5, 1)} "
            f"| {_fmt(decomp['std_ndsm']['median'], 1)} "
            f"| {_fmt(decomp['canopy_var_share']['median'], 2)} |")
    lines += ["", "## Backends vs dsm_hr (deployed calibration workflow)", "",
              "Far out of the backends' training domain; a floor expectation "
              "for Phase 3, not a tuning target.", "",
              "| backend | raw r (median) | cal RMSE m (median) | cal r (median) |",
              "| --- | --- | --- | --- |"]
    for name, agg in summary["backends"].items():
        if agg.get("status") != "ok":
            lines.append(f"| {name} | unavailable: {agg.get('reason', '')} | | |")
            continue
        lines.append(f"| {name} | {_fmt(agg['raw_r']['median'], 2)} "
                     f"| {_fmt(agg['cal_rmse']['median'], 1)} "
                     f"| {_fmt(agg['cal_r']['median'], 2)} |")
    lines += ["", "## Notes", "",
              "- `coarse_dem_offset_corrected` adds the per-tile recorded "
              "`datum_offset_m` (never removed silently; both arms listed).",
              "- `*_zeromean` / `*_affine` are oracle upper bounds (bias or "
              "affine fit estimated on calibration blocks only).",
              "- `perfect_terrain_floor` = dtm + constant canopy mean: the "
              "error a perfect terrain model with no canopy awareness leaves.",
              "- `prior_90m_plus_true_ndsm` = prior_90m + built ndsm: where a "
              "perfect depth model on top of the 90 m prior lands (its "
              "terrain error untouched; oracle, uses reference heights).",
              "- `prior_90m_plus_true_residual` = prior_90m + built residual "
              "(= dsm_hr by construction): RMSE near 0 is a plumbing sanity "
              "check, not a result.",
              "- Datum offsets in forest sites reflect canopy-blur in the "
              "coarse DEM over small clearings, not only the geoid; see "
              "docs/DATA_SOURCES.md.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(hr_dataset.MANIFEST_HR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip-backends", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="also write docs/benchmarks/hr_headroom.md")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "server" / "data"))
    args = parser.parse_args()

    rows, validation = evaluation.load_manifest(args.manifest)
    if validation["errors"]:
        raise SystemExit(f"manifest errors: {validation['errors']}")
    rows = [r for r in rows if r["split"] == args.split]
    rows.sort(key=lambda r: r["tile_id"])
    print(f"[headroom] {len(rows)} {args.split} tiles from {args.manifest}")

    tile_records = [tile_priors(row) for row in rows]
    out_dir = Path(args.out_dir)
    backend_results = ({} if args.skip_backends else
                       run_backends(rows, out_dir / "headroom_cache"))

    now = dt.datetime.now().isoformat(timespec="seconds")
    summary = {
        "generated": now,
        "manifest": str(args.manifest),
        "split": args.split,
        "priors": _prior_aggregates(tile_records),
        "backends": _backend_aggregates(backend_results, tile_records),
        "tile_records": tile_records,
        "backend_tiles": backend_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "headroom_hr.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                    default=str), encoding="utf-8")

    csv_path = out_dir / "headroom_hr.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "arm", "tile_id", "site_id", "stratum",
                         "temporal_flag", "n", "rmse", "mae", "bias", "r"])
        for tile in tile_records:
            for arm_name, arm in sorted(tile["arms"].items()):
                writer.writerow(["prior", arm_name, tile["tile_id"],
                                 tile["site_id"], tile["stratum"],
                                 tile["temporal_flag"], arm.get("n", ""),
                                 _fmt(arm.get("rmse"), 4), _fmt(arm.get("mae"), 4),
                                 _fmt(arm.get("bias"), 4), _fmt(arm.get("r"), 4)])
        for backend, payload in sorted(backend_results.items()):
            for tile_id, arm in sorted(payload["tiles"].items()):
                writer.writerow(["backend", backend, tile_id, "", "", "", arm.get(
                    "n_eval_pixels", ""), _fmt(arm.get("cal_rmse"), 4),
                    _fmt(arm.get("cal_mae"), 4), _fmt(arm.get("cal_bias"), 4),
                    _fmt(arm.get("cal_r"), 4)])
    print(f"[headroom] wrote {json_path} and {csv_path}")

    if args.report:
        report_path = REPO_ROOT / "docs" / "benchmarks" / "hr_headroom.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_md(summary), encoding="utf-8")
        print(f"[headroom] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
