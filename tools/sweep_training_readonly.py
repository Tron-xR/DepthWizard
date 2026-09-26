#!/usr/bin/env python3
"""README-annotated, READ-ONLY diagnostic sweeper for the training-family scorer
results. It never writes to server/ and never invokes production code; it only
calls the same public /evaluate the scorer doesARNING, plus a local numpy
correlation/sweep re-implementation of the scorer's OWN verdict logic so the
diagnostics the user asked for (spatial aggregation + alignment shift) can be
computed against the SAME /evaluate responses WITHOUT changing the scorer.

Phase 2  — raw vs calibrated correlation per tile (already in scorer table).
Phase 3  — spatial aggregation: 1x1, 3x3, 7x7, 15x15, and DEM-native resample.
Phase 4  — alignment sweep: dx,dy in [-5..+5], best shift + corr at zero shift.

This tool exists SOLELY to re-derive and print those diagnostics; it does not
modify scoring semantics and does not touch server code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Re-derived from scorer's own thresholds so the verdicts match scorer's flags.
# ---------------------------------------------------------------------------
RAW_NEG_THRESHOLD = -0.25
CAL_POS_THRESHOLD = 0.25
RAW_WEAK = 0.2

TEMPLATE = "Pearson is sign-invariant under affine calibration ONLY when the "
"scale is positive: corr(s*r + o, g) = sign(s) * corr(r, g). A negative "
"calibrated scale therefore flips a negative raw correlation into a positive "
"calibrated one - the exact downside the scorer's SIGN-INVERTED flag exists "
"to catch (see scorer docstring, tools/score_my_data.py)."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    ap.add_argument("--family", default="hilly", help="hilly|forested|sparse")
    ap.add_argument("--tile", default="00", help="two-digit tile number (00..17)")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--timeout", default=300, type=int)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1] / "training_data"
    rgbs = sorted((root / args.family).glob(f"{args.family}_{args.tile}_rgb.*"))
    dems = sorted((root / args.family).glob(f"{args.family}_{args.tile}_dem.tif"))
    if not (rgbs and dems):
        print(f"No pair for {args.family}_{args.tile} under {root}", file=sys.stderr)
        return 2
    rgb, dem = rgbs[0], dems[0]

    if args.wait:
        for _ in range(10):
            try:
                r = requests.get(f"{base}/health", timeout=2)
                if r.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0)
        else:
            print("server not healthy", file=sys.stderr)
            return 1

    def evaluate(mode: str) -> dict:
        files = {"image": ("rgb.png", rgb.read_bytes(), "image/png"),
                 "dem": (dem.name, dem.read_bytes(), "image/tiff")}
        resp = requests.post(f"{base}/evaluate", files=files,
                             data={"mode": mode}, timeout=args.timeout)
        resp.raise_for_status()
        return resp.json()

    print(f"tile: {args.family}/{rgb.name} + {dem.name}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Reference facts (from /evaluate response, no re-read of rasters here)
    # ------------------------------------------------------------------
    cal = evaluate("calibrated")
    rel = evaluate("relative")
    print("== /evaluate basic facts ==")
    for k in ("valid_pixels", "total_pixels", "held_out_pixel_count"):
        if cal.get(k) is not None:
            print(f"  {k:<26}: {cal[k]}")
    for k in ("correlation", "correlation_reason", "calibration_status",
              "calibration_warning", "calibration_scale", "calibration_offset"):
        if cal.get(k) is not None:
            print(f"  cal.{k:<26}: {cal[k]}")
    print(f"  rel.raw_correlation   : {rel.get('correlation')}")

    # ------------------------------------------------------------------
    # Correlation from the SAME held-out samples, raw vs calibrated
    # (this is exactly what the scorer's ratio row is built from).
    # ------------------------------------------------------------------
    raw_c = rel.get("correlation")
    cal_c = cal.get("correlation")
    print("=" * 100)
    print(f"raw_corr            : {raw_c:+.3f}" if raw_c is not None else "raw_corr            : n/a")
    print(f"calibrated_corr     : {cal_c:+.3f}" if cal_c is not None else "calibrated_corr     : n/a")

    if raw_c is not None and cal_c is not None:
        if raw_c < RAW_NEG_THRESHOLD and cal_c > CAL_POS_THRESHOLD:
            print("VERDICT (scorer flags) : SIGN-INVERTED — raw strongly negative,")
            print("    calibrated flipped POSITIVE. Correlation is sign-invariant")
            print("    under an affine ONLY for positive scale; a negative fitted")
            print("    scale flips the sign. NOT genuine agreement (scorer rule).")
        elif cal_c < 0 and raw_c > 0:
            print("VERDICT (scorer flags) : REVERSED — calibration turned positive")
            print("    raw agreement NEGATIVE.")
        elif raw_c is not None and abs(raw_c) < RAW_WEAK:
            print("VERDICT (scorer flags) : weak-raw — raw correlation too small")
            print("    for any conclusion.")
        else:
            print("VERDICT (scorer flags) : ok")

    # ------------------------------------------------------------------
    # Phase 3 - spatial aggregation: mean over k x k blocks (1,3,7,15) and
    # DEM-native resample. Block means are computed on the PREDICTION and the
    # DEM from the /evaluate relative arrays already in the response.
    # ------------------------------------------------------------------
    rel_arr = rel.get("relative_depth") if isinstance(rel, dict) else None
    dem_arr = rel.get("dem") if isinstance(rel, dict) else None
    if rel_arr is None or dem_arr is None:
        print("    (relative_depth/dem arrays not returned by /evaluate in this")
        print("     deployment - skipping Phase 3/4 aggregation sweeps)")
        return 0

    r = np.asarray(rel_arr, dtype="float64")
    g = np.asarray(dem_arr, dtype="float64")
    mask = np.isfinite(r) & np.isfinite(g)
    if not mask.any():
        print("    (no finite overlap - nothing to aggregate)", file=sys.stderr)
        return 2

    def pearson(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 2:
            return float("nan")
        x = a[m] - np.nanmean(a[m])
        y = b[m] - np.nanmean(b[m])
        denom = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
        return float(np.nansum(x * y) / denom) if denom > 0 else float("nan")

    print("=" * 100)
    print("PHASE 3 — spatial aggregation (block means, prediction vs DEM)")
    print(f"  {'block':<10}{'pearson':>10}{'rmse':>10}{'mae':>10}")
    for k in (1, 3, 7, 15):
        if k == 1:
            rk, gk = r, g
        else:
            n = (r.shape[0] // k) * k
            m = (r.shape[1] // k) * k
            rk = r[:n, :m].reshape(n // k, k, m // k, k).mean(axis=(1, 3))
            gk = g[:n, :m].reshape(n // k, k, m // k, k).mean(axis=(1, 3))
        mm = np.isfinite(rk) & np.isfinite(gk)
        # Calibrated-std units: use RELATIVE rk; metric needs meters, so we
        # apply the reported scale/offset (ABS), not a fresh fit.
        scale = cal.get("calibration_scale")
        off = cal.get("calibration_offset")
        rk_abs = rk * scale + off if scale is not None else rk
        rmse = float(np.sqrt(np.nanmean((rk_abs[mm] - gk[mm]) ** 2)))
        mae = float(np.nanmean(np.abs(rk_abs[mm] - gk[mm])))
        print(f"  {k:>4}x{k:<4}{pearson(rk, gk):>+10.3f}{rmse:>10.2f}{mae:>10.2f}")

    # ------------------------------------------------------------------
    # Phase 4 - alignment sweep on the same held-out relative array.
    # dx,dy in [-5..+5]; report corr at each shift, best + at zero shift.
    # ------------------------------------------------------------------
    print("=" * 100)
    print("PHASE 4 — spatial shift sweep (dx, dy in [-5..+5])")
    print(f"  {'dx':>4}{'dy':>4}{'pearson':>12}{'px_gain':>12}")
    best, best_c = (0, 0), pearson(r, g)
    zero = pearson(r, g)
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx == dy == 0:
                continue
            rshift = np.roll(np.roll(r, dx, axis=1), dy, axis=0)
            c = pearson(rshift, g)
            if not np.isnan(c) and (np.isnan(best_c) or c > best_c):
                best_c, best = c, (dx, dy)
            print(f"  {dx:>4}{dy:>4}{c:>+12.3f}{c - zero:>+12.3f}")
    print(f"best shift     : dx={best[0]:+d}, dy={best[1]:+d}, pearson={best_c:+.3f}")
    print(f"zero shift     : dx=0, dy=0, pearson={zero:+.3f}")
    print(f"alignment gain : {best_c - zero:+.3f}")
    print()
    print(TEMPLATE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
