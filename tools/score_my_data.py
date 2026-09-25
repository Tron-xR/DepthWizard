#!/usr/bin/env python3
"""Batch-scoring harness for own RGB+DEM pairs against the Served /evaluate.

Scores every `{prefix}_{n}_rgb.<ext>` + `{prefix}_{n}_dem.tif` pair found under
a root directory (or explicit pair globs). For each tile we issue TWO /evaluate
calls per the real endpoint contract (routes.py:163, schemas.py:69):

  - mode="relative"    -> raw relative-depth correlation, MAE/RMSE null/default
  - mode="calibrated"  -> affine-calibrated RMSE/MAE + held-out correlation

The sign-inversion hazard from docs/14-evaluation-results.md is surfaced as an
explicit, loud flag rather than a quiet number: a tile whose raw correlation is
clearly negative while its calibrated correlation is clearly positive is marked
SIGN-INVERTED (calibration flipped the sign; the apparent "agreement" is not
genuine). Per-tile failures (server 5xx, malformed geotiff/rgb) are logged and
the batch continues.

Verdicts are reported on TWO independent axes (B3):
  - model polarity    (from the raw signed r): inverted / positive / no-signal
  - calibration stability (raw vs calibrated sign): flipped / stable
A RANSAC sign flip is a calibration artifact and must not masquerade as model
inversion; the legacy single-axis flag is still printed for backward compat.

Run (defaults target training_data/ when the server is up):
  uv run python tools/score_my_data.py --root training_data
  uv run python tools/score_my_data.py --img "geo_test_data/*_rgb.*" --dem "geo_test_data/*_dem.tif"
  uv run python tools/score_my_data.py --root training_data --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import json
import statistics as _stats
import sys
import time
from pathlib import Path

import requests

# --- thresholds for the sign-inversion verdict --------------------------------
# A tile is flagged SIGN-INVERTED when the affine calibration flips the sign of
# the RELATIVE-depth correlation: raw strongly negative (< -0.5) but calibrated
# strongly positive (> +0.5). Calibration is scale+offset only; it cannot invent
# a real positive linear relationship out of a strongly negative one - when it
# appears to, that is the documented degenerate/inverted fit and must be called
# out instead of being presented as agreement.
RAW_NEG_THRESHOLD = -0.25
CAL_POS_THRESHOLD = 0.25


def _pair_key(stem: str, pattern_n: int) -> str:
    """Normalize a raw pair name into a stable per-gestalt label, e.g. hilly_00."""
    return stem


def _raster_summary(rgb: Path, dem: Path) -> dict:
    """Cheap structural facts about the pair: dims/dtype/existence, from headers."""
    out = {"rgb_exists": rgb.exists(), "dem_exists": dem.exists()}
    if not (out["rgb_exists"] and out["dem_exists"]):
        out["reason"] = "missing_side"
        return out
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard server dep
        out["reason"] = "numpy_unavailable"
        return out
    try:
        import rasterio as rio
        with rio.open(rgb) as src:
            out["rgb_shape"] = (src.height, src.width)
            out["rgb_dtype"] = src.dtypes[0] if src.dtypes else None
        with rio.open(dem) as src:
            out["dem_shape"] = (src.height, src.width)
            out["dem_dtype"] = src.dtypes[0] if src.dtypes else None
            out["dem_crs"] = str(src.crs) if src.crs else None
    except Exception:  # noqa: BLE001 - raster open failures are per-tile, not fatal
        out["reason"] = "unreadable_raster"
    return out


def _wait_for_health(base: str, tries: int = 10, delay: float = 1.0) -> bool:
    url = f"{base}/health"
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=2)
            if r.ok and r.json().get("status") == "ok":
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False


def _evaluate(base: str, rgb: Path, dem: Path, mode: str, timeout: int = 300) -> dict:
    """One /evaluate call. Returns the parsed JSON body; raises on transport/HTTP err."""
    files = {
        "image": ("rgb.png", rgb.read_bytes(), "image/png"),
        "dem": (dem.name, dem.read_bytes(), "image/tiff"),
    }
    url = f"{base}/evaluate"
    resp = requests.post(url, files=files, data={"mode": mode}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _safe(key: str, body: dict) -> object:
    return body.get(key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="folder to scan for *_rgb/_*_dem pairs")
    ap.add_argument("--img", default=None, help="explicit glob for RGB sides (e.g. geo_test_data/*_rgb.*)")
    ap.add_argument("--dem", default=None, help="explicit glob for DEM sides (e.g. geo_test_data/*_dem.tif)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    ap.add_argument("--timeout", default=300, type=int)
    ap.add_argument("--no-wait", action="store_true", help="skip the /health preflight")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"

    pairs: list[tuple[str, Path, Path]] = []
    if args.root:
        root = Path(args.root)
        rgb_globs = sorted(root.rglob("*_rgb.*"))
        for rgb in rgb_globs:
            stem = rgb.stem
            if not stem.endswith("_rgb"):
                continue
            dem = rgb.parent / (stem[: -len("_rgb")] + "_dem.tif")
            if dem.exists():
                pairs.append((str(rgb.parent.relative_to(root)), rgb, dem))
    else:
        if not (args.img and args.dem):
            print("Provide --root, or both --img and --dem globs.", file=sys.stderr)
            return 2
        for rgb, dem in zip(_glob.glob(args.img), _glob.glob(args.dem)):
            # pair by stem (xxxxx_rgb.X + xxxxx_dem.tif)
            stem = Path(rgb).stem.replace("_rgb", "")
            pairs.append((stem, Path(rgb), Path(dem)))

    if not pairs:
        print("No RGB+DEM pairs found under the given input.", file=sys.stderr)
        return 2

    print(f"Picked up {len(pairs)} pair(s).")
    for _, rgb, dem in pairs:
        print(f"    {rgb.name}  +  {dem.name}")

    if not args.no_wait and not _wait_for_health(base):
        print(f"Server not healthy at {base}. Start it, then re-run.", file=sys.stderr)
        return 1

    rows = []
    for idx, (prefix, rgb, dem) in enumerate(pairs, start=1):
        label = prefix or rgb.stem
        print(f"\n[{idx}/{len(pairs)}] {label}")
        row = {"tile": label, "raw_corr": None, "cal_corr": None,
               "rmse": None, "mae": None, "held_out": None,
               "flag": "ok", "note": None,
               "polarity": None, "stability": None}
        try:
            raw = _evaluate(base, rgb, dem, "relative", args.timeout)
            cal = _evaluate(base, rgb, dem, "calibrated", args.timeout)
            row["raw_corr"] = _safe("correlation", raw)
            row["cal_corr"] = _safe("correlation", cal)
            row["rmse"] = _safe("rmse", cal)
            row["mae"] = _safe("mae", cal)
            row["held_out"] = _safe("held_out_pixel_count", cal)
        except requests.RequestException as e:
            row["flag"] = "error"
            row["note"] = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {row['note']} (continuing)")
            rows.append(row)
            continue

        raw_c, cal_c = row["raw_corr"], row["cal_corr"]
        if raw_c is not None and cal_c is not None:
            if raw_c < RAW_NEG_THRESHOLD and cal_c > CAL_POS_THRESHOLD:
                row["flag"] = "SIGN-INVERTED"
                row["note"] = ("calibration flipped a clearly negative raw "
                               "correlation into a positive calibrated one")
                print(f"    !! {row['tile']} is SIGN-INVERTED: raw {raw_c:+.3f} -> "
                      f"calibrated {cal_c:+.3f}")
            elif cal_c < 0 and raw_c > 0:
                row["flag"] = "REVERSED"
                row["note"] = "calibrated correlation went NEGATIVE despite raw agreement"
            elif raw_c < 0.2:
                row["flag"] = "weak-raw"
                row["note"] = f"raw correlation {raw_c:+.3f} < 0.2"

        # Two-axis verdict (B3): model polarity from the raw signed r, and
        # calibration stability (did calibration flip the sign?). "flipped"
        # needs a meaningful raw signal (|raw| > 0.05); a missing calibrated
        # correlation is "n/a", never flipped.
        if raw_c is None:
            row["polarity"] = "n/a"
        elif raw_c < RAW_NEG_THRESHOLD:
            row["polarity"] = "inverted"
        elif raw_c > CAL_POS_THRESHOLD:
            row["polarity"] = "positive"
        else:
            row["polarity"] = "no-signal"
        if raw_c is None or cal_c is None:
            row["stability"] = "n/a"
        elif abs(raw_c) <= 0.05:
            row["stability"] = "stable"
        elif (raw_c > 0.0) != (cal_c > 0.0):
            row["stability"] = "flipped"
        else:
            row["stability"] = "stable"
        rows.append(row)

    print()
    print("=" * 128)
    print(f"{'tile':<24}{'raw_corr':>10}{'cal_corr':>10}{'rmse':>10}{'mae':>10}"
          f"{'held_out':>12}   {'flag':<13}{'polarity':>10}{'stability':>10}")
    print("-" * 128)
    for r in rows:
        raw = f"{r['raw_corr']:+.3f}" if r["raw_corr"] is not None else "  n/a"
        cal = f"{r['cal_corr']:+.3f}" if r["cal_corr"] is not None else "  n/a"
        if r["rmse"] is not None:
            rmse = f"{r['rmse']:.2f}"
            mae = f"{r['mae']:.2f}"
        else:
            rmse = "n/a"
            mae = "n/a"
        held = str(r["held_out"]) if r["held_out"] is not None else "n/a"
        flag = r["flag"] or "ok"
        pol = r["polarity"] or "n/a"
        stab = r["stability"] or "n/a"
        print(f"{r['tile']:<24}{raw:>10}{cal:>10}{rmse:>10}{mae:>10}{held:>12}   "
              f"{flag:<13}{pol:>10}{stab:>10}")
        if r.get("note"):
            print(f"    ({r['note']})")

    # aggregates
    rws = [r["raw_corr"] for r in rows if r["raw_corr"] is not None]
    cws = [r["cal_corr"] for r in rows if r["cal_corr"] is not None]
    rms = [r["rmse"] for r in rows if r["rmse"] is not None]
    mae = [r["mae"] for r in rows if r["mae"] is not None]
    sig = [r for r in rows if r["flag"] == "SIGN-INVERTED"]

    print("=" * 112)
    print("AGGREGATES (mean across successfully evaluated tiles)")
    print(f"  raw correlation         : {(_stats.mean(rws) if rws else float('nan')):+.3f}  (n={len(rws)})")
    print(f"  calibrated correlation  : {(_stats.mean(cws) if cws else float('nan')):+.3f}  (n={len(cws)})")
    print(f"  rmse (calibrated)       : {(_stats.mean(rms) if rms else float('nan')):.2f} m  (n={len(rms)})")
    print(f"  mae  (calibrated)       : {(_stats.mean(mae) if mae else float('nan')):.2f} m  (n={len(mae)})")
    print(f"  SIGN-INVERTED tiles     : {len(sig)}")
    for s in sig:
        print(f"      {s['tile']}: raw {s['raw_corr']:+.3f} -> cal {s['cal_corr']:+.3f} (NOT genuine agreement)")
    print("=" * 112)

    # Cross-tab of the two axes (B3): rows = model polarity from the raw signed
    # r; columns = calibration stability (did calibration flip the sign?).
    _pol_order = ["inverted", "positive", "no-signal", "n/a"]
    _stab_order = ["flipped", "stable", "n/a"]
    xt = {(p, s): 0 for p in _pol_order for s in _stab_order}
    for r in rows:
        xt[(r["polarity"], r["stability"])] += 1
    print()
    print("CROSS-TAB: model polarity x calibration stability")
    print("".join(f"{s:>13}" for s in _stab_order))
    print("-" * 62)
    for p in _pol_order:
        cells = "".join(f"{xt[(p, s)]:>13}" for s in _stab_order)
        print(f"{p:>13}{cells}")
    print("-" * 62)
    pos_flip = xt[("positive", "flipped")]
    neg_flip = xt[("inverted", "flipped")]
    print(f"  raw>0 & cal<0  ({pos_flip} tiles): calibration artifact - a negative "
          f"calibrated correlation after a positive raw (expected EMPTY after B1 "
          f"sign-stability fix)")
    print(f"  raw<0 & cal>0  ({neg_flip} tiles): expected flip for an inverted "
          f"model - calibration honestly re-inverts it; the calibrated positive "
          f"r is not genuine agreement")
    annotated_flip = [r for r in rows if r["stability"] == "flipped"]
    if annotated_flip:
        print("  flipped tiles:")
        for f_ in annotated_flip:
            print(f"      {f_['tile']}: polarity={f_['polarity']:>9} raw {f_['raw_corr']:+.3f} "
                  f"-> cal {f_['cal_corr']:+.3f}")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
