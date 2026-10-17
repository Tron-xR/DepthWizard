#!/usr/bin/env python3
"""Root-cause audit for DepthWizard's geographic generalization.

Read-only analysis of the CACHED benchmark predictions (no model reruns) plus
an optional small GPU probe of the fine-tuned backend's train/inference parity.

analyze:
    Joins benchmark per_tile.csv metrics (protocols A/B) with new per-tile
    prediction statistics computed from the cached .npy predictions + DEMs:
    rDSM std/saturation, DEM relief, recomputed raw correlation (consistency
    check against the benchmark), coarse/fine (31 px) correlation split,
    per-tile pixel-shift alignment tests
    (prediction vs DEM, +-16 px, phase-random null, and transpose),
    cross-backend agreement (whole/coarse/fine), 
    near-constant counts, polarity mix by region/stratum, the fine-tuned
    train/val split separation, and protocol-B frozen-transfer vs the
    constant dev-median baseline per tile.

probe:
    Fresh inference (GPU) on a few tiles: DA2-small backbone depth statistics
    at native resolution (predicted_depth comes out at a 14-multiple size,
    e.g. 714² for a 720² input), clamp(0,50) effect on the head, the
    production-mirror LANCZOS resize of the head output back to native size,
    and the same-area 256-crop training-scale comparison (depth value
    distribution shifts strongly with input size). Writes probe.json into the
    analyze run dir.

Writes ONLY into a NEW dir: benchmark_runs/<commit7>_generalization_audit_<utc>Z/

Usage (from server/):
  uv run python ../tools/audit_generalization.py analyze
  uv run python ../tools/audit_generalization.py analyze --selfcheck
  uv run python ../tools/audit_generalization.py probe --run-dir <analyze dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "benchmark_runs"
TILES_ROOT = REPO_ROOT / "training_data"
CKPT = REPO_ROOT / "decoder_training" / "checkpoints" / "da2_decoder_finetuned.pt"

# Canonical runs (chosen in the audit; each holds all 54-tile predictions).
RUNS_A = {  # protocol A: per-tile 80/20 pixel split, historical pipeline
    "pix2pix": "c3a2841_pix2pix_20261016T134335Z",
    "finetuned": "c3a2841_finetuned_20261016T134947Z",
    "imele": "c3a2841_imele_20261016T155949Z",
    "da2": "c3a2841_da2_20261016T162855Z",
}
RUNS_B = {  # protocol B: frozen dev fit, test regions only
    "pix2pix": "c3a2841_pix2pix_20261016T134627Z",
    "finetuned": "c3a2841_finetuned_20261016T135223Z",
    "imele": "c3a2841_imele_20261016T162713Z",
    "da2": "c3a2841_da2_20261016T163205Z",
}
BACKENDS = list(RUNS_A)
MANIFEST = BENCH / RUNS_A["pix2pix"] / "manifest.csv"

RAW_NEG = -0.25   # mirrors tools/benchmark_tiles.py flag thresholds
RAW_WEAK = 0.2
SHIFT_MAX = 16
SHIFT_CROP = 512


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float64").ravel()
    b = np.asarray(b, dtype="float64").ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _phase_randomize(arr: np.ndarray, rng) -> np.ndarray:
    """Same magnitude spectrum, random phases: null field with no alignment.

    Phases come from FFT-ing a random real field, so the DC/Nyquist bins stay
    real (the null keeps the original mean level) and the output is exactly
    real."""
    f = np.fft.rfft2(arr)
    ph = np.fft.rfft2(rng.rand(*arr.shape))
    return np.fft.irfft2(np.abs(f) * np.exp(1j * np.angle(ph)), s=arr.shape)


def _lowpass(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    return uniform_filter(a, 31)


def _highpass(a: np.ndarray) -> np.ndarray:
    return a - _lowpass(a)


def _read_csv_map(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["tile_id"]: r for r in csv.DictReader(f)}


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _rdsm_stats(r: np.ndarray) -> dict:
    lo, hi = float(r.min()), float(r.max())
    span = hi - lo
    sat = float(((r <= lo + 1e-6) | (r >= hi - 1e-6)).mean()) if span > 0 else 1.0
    return {"std": float(r.std()), "min": lo, "max": hi, "sat_frac": sat}


def best_shift(pred: np.ndarray, dem: np.ndarray,
               max_shift: int = SHIFT_MAX, crop: int = SHIFT_CROP) -> dict:
    """Best integer (dx, dy) aligning pred(y,x) with dem(y+dy, x+dx).

    Both windows are high-passed first (31 px uniform filter): raw terrain has
    a strong low-frequency ramp that makes corr() increase monotonically with
    any shift, pinning a naive search at its boundary. The peak is located by
    FFT cross-correlation over +-max_shift, then real Pearson at the peak,
    at zero shift (r0) and a transpose canary are computed on the high-pass."""
    from scipy.ndimage import uniform_filter

    h, w = pred.shape
    if min(h, w) < crop + 2 * max_shift:
        crop = min(h, w) - 2 * max_shift
    cy, cx = h // 2, w // 2
    y0, x0 = cy - crop // 2, cx - crop // 2
    p = pred[y0:y0 + crop, x0:x0 + crop].astype("float64")
    p = p - uniform_filter(p, 31)
    d = dem[y0 - max_shift:y0 + crop + max_shift,
            x0 - max_shift:x0 + crop + max_shift].astype("float64")
    d = d - uniform_filter(d, 31)
    d0 = d[max_shift:max_shift + crop, max_shift:max_shift + crop]
    from scipy.signal import correlate
    # correlate(d, p, 'valid')[k0,k1] = sum_n d(n+k) p(n): peak index minus
    # max_shift is exactly (dy, dx) for pred(y,x) vs dem(y+dy, x+dx)
    cc = correlate(d, p, mode="valid", method="fft")
    k0, k1 = np.unravel_index(int(np.argmax(cc)), cc.shape)
    dy, dx = int(k0) - max_shift, int(k1) - max_shift
    r_best = _pearson(p, d[max_shift + dy:max_shift + dy + crop,
                           max_shift + dx:max_shift + dx + crop])
    r0 = _pearson(p, d0)
    return {"dx": int(dx), "dy": int(dy), "r_best": round(r_best, 6),
            "r0": round(r0, 6), "gain": round(r_best - r0, 6),
            "r_transpose": round(_pearson(p, d0.T), 6)}


def _load_pred(run: str, tile_id: str):
    p = BENCH / run / "predictions" / f"{tile_id}_relative.npy"
    return np.load(p).astype("float64") if p.exists() else None


def _load_dem(stratum: str, tile_id: str):
    import rasterio

    with rasterio.open(TILES_ROOT / stratum / f"{tile_id}_dem.tif") as src:
        dem = src.read(1).astype("float64")
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        dem = np.where(dem == nodata, np.nan, dem)
    return dem


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return round(float(np.mean(xs)), 6) if xs else None


def _git(*cmd) -> str:
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def analyze(out_dir: Path) -> dict:
    manifest = _read_csv_map(MANIFEST)
    tiles = sorted(manifest)
    per_a = {b: _read_csv_map(BENCH / RUNS_A[b] / "per_tile.csv") for b in BACKENDS}
    per_b = {b: _read_csv_map(BENCH / RUNS_B[b] / "per_tile.csv")
             for b in BACKENDS if (BENCH / RUNS_B[b] / "per_tile.csv").exists()}

    dems, preds = {}, {}
    for t in tiles:
        dems[t] = _load_dem(manifest[t]["stratum"], t)
        preds[t] = {b: _load_pred(RUNS_A[b], t) for b in BACKENDS}

    rows = []
    rng = np.random.RandomState(0)
    raw_delta_max = 0.0
    shape_errors = []
    for t in tiles:
        m = manifest[t]
        dem = dems[t]
        valid = np.isfinite(dem)
        row = {
            "tile_id": t, "stratum": m["stratum"], "region": m["region"],
            "protocol_b_split": m["protocol_b_split"],
            "finetuned_split": m["finetuned_split"],
            "dem_source": "tilezen" if m["crs"] == "EPSG:3857" else "cop30",
            "width": m["width"], "height": m["height"], "crs": m["crs"],
            "dem_std": round(float(np.nanstd(dem)), 4),
            "dem_range": round(float(np.nanmax(dem) - np.nanmin(dem)), 4),
        }
        for b in BACKENDS:
            r = preds[t][b]
            if r is None or r.shape != dem.shape:
                shape_errors.append(f"{b}:{t}:{None if r is None else r.shape}")
                continue
            raw = _pearson(r[valid], dem[valid])
            ref = _f(per_a[b][t]["raw_corr"]) if t in per_a[b] else None
            if ref is not None:
                raw_delta_max = max(raw_delta_max, abs(raw - ref))
            st = _rdsm_stats(r)
            row[f"{b}_raw_corr"] = round(raw, 6)
            row[f"{b}_rdsm_std"] = round(st["std"], 6)
            row[f"{b}_rdsm_sat"] = round(st["sat_frac"], 6)
            row[f"{b}_rdsm_dem_std_ratio"] = (
                round(st["std"] / row["dem_std"], 6) if row["dem_std"] else None)
            row[f"{b}_coarse_corr"] = round(
                _pearson(_lowpass(r)[valid], _lowpass(dem)[valid]), 6)
            row[f"{b}_fine_corr"] = round(
                _pearson(_highpass(r)[valid], _highpass(dem)[valid]), 6)
            a = per_a[b].get(t)
            if a:
                row[f"{b}_cal_corr"] = _f(a.get("cal_corr"))
                row[f"{b}_rmse"] = _f(a.get("rmse"))
                row[f"{b}_mae"] = _f(a.get("mae"))
                row[f"{b}_scale"] = _f(a.get("calibration_scale"))
                warn = a.get("calibration_warning") or ""
                row[f"{b}_near_constant"] = int("prediction_near_constant" in warn)
                row[f"{b}_flag"] = a.get("flag")
            sh = best_shift(r, dem)
            null_gain = max(
                best_shift(r, _phase_randomize(dem, rng))["gain"] for _ in range(2))
            row[f"{b}_shift_dx"] = sh["dx"]
            row[f"{b}_shift_dy"] = sh["dy"]
            row[f"{b}_r0"] = sh["r0"]
            row[f"{b}_r_best"] = sh["r_best"]
            row[f"{b}_shift_gain"] = sh["gain"]
            row[f"{b}_shift_null_max"] = round(null_gain, 6)
            row[f"{b}_shift_evidence"] = int(sh["gain"] > null_gain + 0.05)
            row[f"{b}_r_transpose"] = sh["r_transpose"]
            bm = per_b.get(b, {}).get(t)
            if bm:
                row[f"{b}_b_rmse"] = _f(bm.get("rmse"))
                row[f"{b}_b_baseline_rmse"] = _f(bm.get("baseline_rmse"))
                r1, r2 = _f(bm.get("rmse")), _f(bm.get("baseline_rmse"))
                row[f"{b}_b_model_better"] = (
                    int(r1 < r2) if r1 is not None and r2 is not None else None)
        rows.append(row)

    summary = _aggregate(rows, per_a, per_b, tiles, preds)
    summary["consistency"] = {"raw_corr_recompute_max_abs_delta": raw_delta_max,
                              "shape_errors": shape_errors}
    summary["provenance"] = {
        "analyze": "cached predictions only (no fresh model inference)",
        "runs_a": RUNS_A, "runs_b": RUNS_B,
        "note": "probe.json (if present) contains the only fresh inference.",
    }

    _write(out_dir, rows, summary)
    return summary


def _aggregate(rows, per_a, per_b, tiles, preds) -> dict:
    out: dict = {"n_tiles": len(tiles), "backends": {}}
    for b in BACKENDS:
        have = [r for r in rows if f"{b}_raw_corr" in r]
        raws = [r[f"{b}_raw_corr"] for r in have]
        out["backends"][b] = {
            "n": len(have),
            "mean_raw_corr": _mean(raws),
            "median_raw_corr": (round(float(np.median(raws)), 6) if raws else None),
            "frac_inverted": _frac(raws, lambda x: x <= RAW_NEG),
            "frac_weak": _frac(raws, lambda x: RAW_NEG < x < RAW_WEAK),
            "frac_ok": _frac(raws, lambda x: x >= RAW_WEAK),
            "mean_abs_raw_corr": _mean([abs(x) for x in raws]),
            "mean_cal_corr": _mean([_f(per_a[b][t].get("cal_corr")) for t in per_a[b]]),
            "mean_rmse": _mean([_f(per_a[b][t].get("rmse")) for t in per_a[b]]),
            "mean_mae": _mean([_f(per_a[b][t].get("mae")) for t in per_a[b]]),
            "near_constant": sum(
                1 for t in per_a[b]
                if "prediction_near_constant" in (per_a[b][t].get("calibration_warning") or "")),
            "polarity_mix_by_stratum": _group(raws, [r for r in have], "stratum"),
            "polarity_mix_by_region": _group(raws, [r for r in have], "region"),
            "mean_rdsm_std": _mean([r.get(f"{b}_rdsm_std") for r in have]),
            "mean_coarse_corr": _mean([r.get(f"{b}_coarse_corr") for r in have]),
            "mean_fine_corr": _mean([r.get(f"{b}_fine_corr") for r in have]),
            "mean_fine_corr_by_stratum": {
                k: _mean([r.get(f"{b}_fine_corr") for r in have
                          if r["stratum"] == k])
                for k in sorted({r["stratum"] for r in have})},
            "shift_evidence_count": sum(
                1 for r in have if r.get(f"{b}_shift_evidence")),
            "max_abs_shift_among_evidence": max(
                [max(abs(r.get(f"{b}_shift_dx") or 0), abs(r.get(f"{b}_shift_dy") or 0))
                 for r in have if r.get(f"{b}_shift_evidence")], default=0),
            "median_r0": (round(float(np.median(
                [r[f"{b}_r0"] for r in have if r.get(f"{b}_r0") is not None])), 6)
                if have else None),
            "transpose_evidence_count": sum(
                1 for r in have
                if (r.get(f"{b}_r_transpose") or -2) - (r.get(f"{b}_r_best") or -2) > 0.05),
        }
        # protocol B frozen transfer vs constant baseline
        bt = [r for r in rows if f"{b}_b_rmse" in r]
        if bt:
            better = [r for r in bt if r[f"{b}_b_model_better"] == 1]
            out["backends"][b]["protocol_b"] = {
                "n_test_tiles": len(bt),
                "model_better_count": len(better),
                "model_worse_count": len(bt) - len(better),
                "mean_model_rmse": _mean([r[f"{b}_b_rmse"] for r in bt]),
                "mean_baseline_rmse": _mean([r[f"{b}_b_baseline_rmse"] for r in bt]),
                "mean_frozen_cal_corr": _mean([_f(per_b[b].get(r["tile_id"], {}).get("cal_corr"))
                                               for r in bt]),
                "by_region": _baseline_by_region(bt, b),
            }

    # cross-backend agreement (per-tile Pearson between rDSMs)
    pair = {}
    for i, b1 in enumerate(BACKENDS):
        for b2 in BACKENDS[i + 1:]:
            vals, cvals, fvals = [], [], []
            for t in tiles:
                p1, p2 = preds[t][b1], preds[t][b2]
                if p1 is not None and p2 is not None and p1.shape == p2.shape:
                    vals.append(_pearson(p1, p2))
                    cvals.append(_pearson(_lowpass(p1), _lowpass(p2)))
                    fvals.append(_pearson(_highpass(p1), _highpass(p2)))
            pair[f"{b1}::{b2}"] = {"mean_corr": _mean(vals),
                                   "mean_coarse_corr": _mean(cvals),
                                   "mean_fine_corr": _mean(fvals),
                                   "n": len(vals)}
    out["cross_backend_agreement"] = pair

    # fine-tuned split separation (protocol A, all 54 tiles)
    ft = [r for r in rows if "finetuned_raw_corr" in r]
    out["finetuned_split"] = {}
    for split in ("train", "val"):
        sub = [r for r in ft if r["finetuned_split"] == split]
        out["finetuned_split"][split] = {
            "n": len(sub),
            "mean_raw_corr": _mean([r["finetuned_raw_corr"] for r in sub]),
            "mean_cal_corr": _mean([_f(per_a["finetuned"][r["tile_id"]].get("cal_corr"))
                                    for r in sub]),
            "frac_inverted": _frac([r["finetuned_raw_corr"] for r in sub],
                                   lambda x: x <= RAW_NEG),
        }
    return out


def _frac(xs, pred) -> float | None:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return round(float(np.mean([pred(x) for x in xs])), 4) if xs else None


def _group(raws, rows, key) -> dict:
    groups: dict = {}
    for r, x in zip(rows, raws):
        g = groups.setdefault(r[key], {"n": 0, "vals": []})
        g["n"] += 1
        g["vals"].append(x)
    return {k: {"n": v["n"], "mean_raw_corr": _mean(v["vals"]),
                "frac_inverted": _frac(v["vals"], lambda x: x <= RAW_NEG)}
            for k, v in sorted(groups.items())}


def _baseline_by_region(rows, b) -> dict:
    groups: dict = {}
    for r in rows:
        g = groups.setdefault(r["region"], {"n": 0, "better": 0, "delta": []})
        g["n"] += 1
        g["better"] += int(r[f"{b}_b_model_better"] == 1)
        g["delta"].append(r[f"{b}_b_rmse"] - r[f"{b}_b_baseline_rmse"])
    return {k: {"n": v["n"], "model_better": v["better"],
                "mean_rmse_delta": _mean(v["delta"])}
            for k, v in sorted(groups.items())}


def _write(out_dir: Path, rows, summary) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    cols = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out_dir / "audit_per_tile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    (out_dir / "run.json").write_text(json.dumps({
        "argv": sys.argv,
        "git_commit": _git("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_git("git", "status", "--porcelain")),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    print(f"wrote {out_dir}")


def _summary_md(s: dict) -> str:
    lines = ["# Generalization audit — summary", "",
             f"Tiles: {s['n_tiles']}  (cached predictions only; probe separate)", ""]
    lines += ["| backend | n | mean raw r | |r| mean | inverted | weak | ok | near-const | mean cal r | mean RMSE | coarse r | fine r |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, v in s["backends"].items():
        lines.append(
            f"| {b} | {v['n']} | {v['mean_raw_corr']} | {v['mean_abs_raw_corr']} | "
            f"{v['frac_inverted']} | {v['frac_weak']} | {v['frac_ok']} | "
            f"{v['near_constant']} | {v['mean_cal_corr']} | {v['mean_rmse']} | "
            f"{v['mean_coarse_corr']} | {v['mean_fine_corr']} |")
    lines += ["", "## Protocol B (frozen dev fit vs constant baseline)", ""]
    lines += ["| backend | test tiles | model better | mean model RMSE | mean baseline RMSE |",
              "|---|---|---|---|---|"]
    for b, v in s["backends"].items():
        pb = v.get("protocol_b")
        if pb:
            lines.append(f"| {b} | {pb['n_test_tiles']} | {pb['model_better_count']} | "
                         f"{pb['mean_model_rmse']} | {pb['mean_baseline_rmse']} |")
    lines += ["", "## Alignment (high-passed FFT shift test, ±16 px, center 512²)", ""]
    lines += ["| backend | shift evidence (gain > phase-random null) | max |shift| | transpose evidence | median r at 0 |",
              "|---|---|---|---|---|"]
    for b, v in s["backends"].items():
        lines.append(f"| {b} | {v['shift_evidence_count']} | "
                     f"{v['max_abs_shift_among_evidence']} | "
                     f"{v['transpose_evidence_count']} | {v['median_r0']} |")
    lines += ["", "## Cross-backend rDSM agreement (mean per-tile Pearson: whole / coarse / fine)", ""]
    for k, v in s["cross_backend_agreement"].items():
        lines.append(f"- {k}: {v['mean_corr']} / {v['mean_coarse_corr']} / "
                     f"{v['mean_fine_corr']} (n={v['n']})")
    lines += ["", "## Fine-scale corr vs DEM by stratum", ""]
    strata = ("forested", "hilly", "sparse")
    lines += ["| backend | " + " | ".join(strata) + " |", "|---" * (len(strata) + 1) + "|"]
    for b, v in s["backends"].items():
        fs = v.get("mean_fine_corr_by_stratum", {})
        lines.append(f"| {b} | " + " | ".join(str(fs.get(k)) for k in strata) + " |")
    lines += ["", "## Fine-tuned split separation (protocol A)", ""]
    for k, v in s["finetuned_split"].items():
        lines.append(f"- {k}: n={v['n']} raw_r={v['mean_raw_corr']} "
                     f"cal_r={v['mean_cal_corr']} inverted={v['frac_inverted']}")
    c = s["consistency"]
    lines += ["", f"Recomputed-vs-benchmark raw corr max |delta|: "
                  f"{c['raw_corr_recompute_max_abs_delta']:.2e}",
              f"Shape errors: {c['shape_errors'] or 'none'}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# probe (fresh inference, fine-tuned backend only)
# ---------------------------------------------------------------------------
def probe(run_dir: Path, tiles: list[str]) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "server"))
    import rasterio
    import torch

    from app.pipeline import decoder_head, depth as depth_mod

    manifest = _read_csv_map(MANIFEST)
    backend = depth_mod._FinetunedDepthBackend(str(CKPT))
    results = {}

    for t in tiles:
        m = manifest[t]
        with rasterio.open(TILES_ROOT / m["stratum"] / f"{t}_rgb.tif") as src:
            rgb = src.read().astype("uint8").transpose(1, 2, 0)
        dem = _load_dem(m["stratum"], t)
        h, w = rgb.shape[:2]

        with torch.no_grad():
            pv = decoder_head.to_pixel_values(rgb, backend.proc).to(backend.device)
            dep = backend.model(pixel_values=pv.unsqueeze(0)).predicted_depth
            out_noclamp = backend.head(dep)[0, 0].cpu().numpy().astype("float32")
            out_clamp = backend.head(dep.clamp(0, 50))[0, 0].cpu().numpy().astype("float32")

            cy, cx = h // 2, w // 2
            crop = np.ascontiguousarray(rgb[cy - 128:cy + 128, cx - 128:cx + 128])
            pvc = decoder_head.to_pixel_values(crop, backend.proc).to(backend.device)
            depc = backend.model(pixel_values=pvc.unsqueeze(0)).predicted_depth
            out_crop = backend.head(depc.clamp(0, 50))[0, 0].cpu().numpy().astype("float32")

        from PIL import Image as PILImage

        cdem = dem[cy - 128:cy + 128, cx - 128:cx + 128]
        head_shape = list(out_noclamp.shape)

        def _to_native(arr, size):
            if arr.shape == size:
                return arr
            return np.asarray(PILImage.fromarray(arr).resize((size[1], size[0]),
                                                            PILImage.LANCZOS),
                              dtype="float32")

        # mirror _FinetunedDepthBackend: head output resized back to native size
        out_noclamp = _to_native(out_noclamp, (h, w))
        out_clamp = _to_native(out_clamp, (h, w))
        out_crop = _to_native(out_crop, cdem.shape)

        d = dep.detach().cpu().numpy().ravel()
        dc = depc.detach().cpu().numpy().ravel()
        valid = np.isfinite(dem)
        results[t] = {
            "shape": [h, w],
            "backbone_depth_shape": list(dep.shape[-2:]),
            "head_output_shape": head_shape,
            "backbone_depth_native": {
                "min": float(d.min()), "max": float(d.max()),
                "mean": float(d.mean()), "p1": float(np.percentile(d, 1)),
                "p99": float(np.percentile(d, 99)),
                "frac_gt_50": float((d > 50).mean()),
                "frac_lt_0": float((d < 0).mean()),
            },
            "backbone_depth_256crop": {
                "min": float(dc.min()), "max": float(dc.max()),
                "mean": float(dc.mean()), "frac_gt_50": float((dc > 50).mean()),
            },
            "head_output": {
                "corr_noclamp_vs_clamp": _pearson(out_noclamp, out_clamp),
                "corr_noclamp_vs_dem": _pearson(out_noclamp[valid], dem[valid]),
                "corr_clamp_vs_dem": _pearson(out_clamp[valid], dem[valid]),
                "corr_crophead_vs_demcrop": _pearson(out_crop, cdem),
                "noclamp_std": float(out_noclamp.std()),
                "clamp_std": float(out_clamp.std()),
                "crop_clamp_std": float(out_crop.std()),
            },
            "fresh_inference": True,
        }
        print(t, json.dumps(results[t], indent=None))

    out = run_dir / "probe.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return results


# ---------------------------------------------------------------------------
def _selfcheck() -> None:
    rng = np.random.RandomState(0)
    base = rng.rand(600, 600).astype("float64")
    for _ in range(20):
        base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0)
                + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5.0
    base = 0.8 * base + 0.2 * rng.rand(600, 600)   # keep fine structure
    pred = np.roll(base, 2, axis=0)          # pred(y) == dem(y-2) -> dy=-2
    sh = best_shift(pred, base)
    assert (sh["dx"], sh["dy"]) == (0, -2), sh
    assert sh["gain"] > 0.05, sh
    pred2 = np.roll(base, -3, axis=1)        # pred(x) == dem(x+3) -> dx=+3
    sh2 = best_shift(pred2, base)
    assert (sh2["dx"], sh2["dy"]) == (3, 0), sh2
    null_g = best_shift(pred, _phase_randomize(base, np.random.RandomState(1)))["gain"]
    assert null_g < sh["gain"], (null_g, sh)
    assert _pearson(base, base) > 0.999999
    st = _rdsm_stats(np.linspace(0, 1, 101).reshape(101, 1))
    assert abs(st["std"] - 0.2915) < 0.01 and st["sat_frac"] == 2 / 101, st
    print("selfcheck OK:", sh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--out", type=str, default="")
    a.add_argument("--selfcheck", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--tiles", type=str,
                   default="hilly_06,hilly_10,forested_00,sparse_00")
    args = ap.parse_args()

    if args.cmd == "analyze":
        if args.selfcheck:
            _selfcheck()
            return
        if args.out:
            out_dir = Path(args.out)
        else:
            commit = _git("git", "rev-parse", "--short=7", "HEAD") or "unknown"
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = BENCH / f"{commit}_generalization_audit_{ts}"
        t0 = time.time()
        s = analyze(out_dir)
        print(f"analyze done in {time.time() - t0:.1f}s")
        print(json.dumps({"backends": {b: {k: v[k] for k in
                     ("mean_raw_corr", "frac_inverted", "frac_weak", "frac_ok",
                      "near_constant")} for b, v in s["backends"].items()}}, indent=2))
    else:
        probe(Path(args.run_dir), args.tiles.split(","))


if __name__ == "__main__":
    main()
