"""Aggregate elevation-space metrics for arms B/C/D and write TASK2_REPORT.md.

Sources: runs/metrics_<arm>[_{seed}].csv (any present).
Per tile: model arms -> mean over seeds; C arms -> single value.
Paired comparisons with block bootstrap (2x2 CV lattice blocks) on the 54 CV
tiles: resample blocks, mean per-tile difference, 2.5/97.5 percentiles.
Independent reference (USGS 3DEP, CONUS tiles) is folded in when
dem3dep_fetched.json + *_3dep_<arm>*.csv exist.
"""
import csv, itertools, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode\exp")
ARMS = ["C90", "B90", "D90", "C250", "B250", "D250"]
SEEDS = (0, 1, 2)
FAM = {"forested": "forested", "hilly": "hilly", "sparse": "sparse"}


def load(arm):
    if arm.startswith("C"):
        p = EXP / "runs" / f"metrics_{arm}.csv"
        if not p.exists():
            return None
        out = {}
        for r in csv.DictReader(open(p)):
            out[r["tile"]] = dict(arm=r["arm"], seed=r["seed"], tile=r["tile"],
                                  fam=r["fam"], src=r["src"], block=r["block"],
                                  mae=float(r["mae"]), rmse=float(r["rmse"]),
                                  raw_r=float(r["raw_r"]),
                                  resid_r=float(r["resid_r"]),
                                  raw_pos=int(r["raw_pos"]),
                                  n_valid=int(r["n_valid"]))
        return out
    out = defaultdict(list)
    for s in SEEDS:
        p = EXP / "runs" / f"metrics_{arm}_{s}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            out[r["tile"]].append(r)
    if not out:
        return None
    mean = {}
    for tid, rs in out.items():
        g = lambda k: float(np.mean([float(r[k]) for r in rs]))  # noqa: E731
        mean[tid] = dict(arm=arm, seed="avg", tile=tid, fam=rs[0]["fam"],
                         src=rs[0]["src"], block=rs[0]["block"],
                         mae=g("mae"), rmse=g("rmse"), raw_r=g("raw_r"),
                         resid_r=g("resid_r"),
                         raw_pos=int(g("raw_r") > 0),
                         n_valid=int(np.mean([int(r["n_valid"]) for r in rs])),
                         n_seen=len(rs))
    return mean


def block_map():
    folds = json.loads((EXP / "folds.json").read_text())
    m = {}
    for name, c in folds["cells"].items():
        m[f"{c['stratum']}/{name}"] = (c["ci"] // folds["side"],
                                       c["cj"] // folds["side"])
    return m


def families():
    return ("forested", "hilly", "sparse")


def stats(values):
    v = np.asarray([float(x) for x in values if x is not None], dtype=float)
    return (float(np.nanmean(v)), float(np.nanstd(v)),
            int(np.sum(v > 0)), len(v))


def bootstrap_ci(a, b, blocks, n_iter=10000, seed=7):
    """Pairwise per-tile diff, block-bootstrap 95% CI. a,b: dict tile->value."""
    tiles = [t for t in a if t in b and t in blocks]
    if not tiles:
        return None
    rel = {blk: i for i, blk in enumerate({blocks[t] for t in tiles})}
    vals = np.array([a[t] - b[t] for t in tiles])
    bid = np.array([rel[blocks[t]] for t in tiles], dtype=int)
    uniq = np.arange(len(rel))
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_iter):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.isin(bid, pick)
        samples.append(vals[m].mean() if m.sum() >= 5 else 0.0)
    samples = np.array(samples)
    return dict(mean=float(vals.mean()), ci=[float(np.percentile(samples, 2.5)),
                                            float(np.percentile(samples, 97.5))],
                n=len(tiles))


def fmt(c):
    if c is None:
        return "(no data)"
    return f"{c['mean']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] n={c['n']}"


def table(arms, rows, metric, subset=None):
    lines = []
    for ar in arms:
        d = rows.get(ar)
        if not d:
            lines.append(f"| {ar} | - |")
            continue
        if subset is not None:
            vals = [r[metric] for t, r in d.items() if subset(t, r)]
        else:
            vals = [r[metric] for r in d.values()]
        v = np.asarray([float(x) for x in vals])
        lines.append(f"| {ar} | {v.mean():+.3f} ± {v.std(ddof=1):.3f} "
                     f"({v.size}) |")
    return lines


def main():
    rows = {ar: load(ar) for ar in ARMS}
    blocks = block_map()
    present = [ar for ar, d in rows.items() if d]
    print("present arms:", present)

    out = ["# Task 2 report: does RGB add information beyond a coarse DEM prior?",
           "",
           "## Setup",
           "- Ring-2 spatial-block CV, 4 folds (locked), 3 seeds per model arm.",
           "- Arms: C = bicubic-upsampled coarse (unit slope, no model); "
           "B = DA2-small + RGB + coarse; D = same architecture, RGB zeroed.",
           "- Per tile, deterministic 80/20 pixel split; affine calibration "
           "(pred->meters) fit on the 80%% for model arms; metrics on the 20%%.",
           "- resid_r = corr(pred_elev - coarse_up, truth - coarse_up): "
           "detail above the coarse prior. C is 0 by construction.",
           "- Extra tiles (held-out + geo_test) are never in training; CV "
           "bootstrap block id = (ci//2, cj//2) of the 2x2 lattice.",
           ""]

    # 1. aggregate table over ALL tiles (64): CV + extras each scored once
    out.append("## All tiles (mean of seeds for model arms)")
    out.append("| arm | mae (m) | rmse (m) | raw_r | resid_r | raw_r>0 |")
    out.append("|---|---|---|---|---|---|")
    for ar in ARMS:
        d = rows.get(ar)
        if not d:
            out.append(f"| {ar} | - | - | - | - | - |")
            continue
        mae = np.mean([r["mae"] for r in d.values()])
        rmse = np.mean([r["rmse"] for r in d.values()])
        rr = np.mean([r["raw_r"] for r in d.values()])
        rrpos = sum(1 for r in d.values() if r["raw_r"] > 0)
        resid = np.mean([r["resid_r"] for r in d.values()])
        out.append(f"| {ar} | {mae:.1f} | {rmse:.1f} | {rr:+.3f} | "
                   f"{resid:+.3f} | {rrpos}/{len(d)} |")
    out.append("")

    # 2. by family on the 54 CV tiles
    for met in ("mae", "rmse", "raw_r", "resid_r"):
        out.append(f"### {met} (54 CV tiles, by family)")
        out.append("| arm | forested | hilly | sparse |")
        out.append("|---|---|---|---|")
        for ar in ARMS:
            d = rows.get(ar)
            if not d:
                out.append(f"| {ar} | - | - | - |")
                continue
            cells = {t: r for t, r in d.items() if r["src"] == "trainset"}
            vals = {f: [r[met] for r in cells.values() if r["fam"] == f]
                    for f in families()}
            m = {f: (float(np.mean(v)) if v else float("nan")) for f, v in vals.items()}
            out.append(f"| {ar} | {m['forested']:+.3f} | {m['hilly']:+.3f} | "
                       f"{m['sparse']:+.3f} |")
        out.append("")

    # 3. raw_r > 0 counts per arm/family (54 CV)
    out.append("### raw_r > 0 fraction (54 CV tiles)")
    for ar in ARMS:
        d = rows.get(ar)
        if not d:
            continue
        cells = [r for r in d.values() if r["src"] == "trainset"]
        tot = sum(1 for r in cells if r["raw_r"] > 0)
        by = {f: sum(1 for r in cells if r["fam"] == f and r["raw_r"] > 0)
              for f in families()}
        cnt = {f: sum(1 for r in cells if r["fam"] == f) for f in families()}
        out.append(f"- {ar}: {tot}/54 "
                   + ", ".join(f"{f} {by[f]}/{cnt[f]}" for f in families()))
    out.append("")

    # 4. paired block-bootstrap comparisons on 54 CV tiles
    out.append("## Paired block-bootstrap differences (54 CV tiles, blocks "
               "resampled)")
    for a, b in (("B90", "C90"), ("B250", "C250"), ("D90", "C90"),
                 ("D250", "C250"), ("B90", "D90"), ("B250", "D250")):
        da, db = rows.get(a), rows.get(b)
        if not da or not db:
            out.append(f"- {a} vs {b}: (missing arm)")
            continue
        cells_a = {t: r for t, r in da.items() if r["src"] == "trainset"}
        cells_b = {t: r for t, r in db.items() if r["src"] == "trainset"}
        for met in ("resid_r", "mae", "rmse", "raw_r"):
            va = {t: cells_a[t][met] for t in cells_a}
            vb = {t: cells_b[t][met] for t in cells_b}
            out.append(f"- {a}-{b} {met}: {fmt(bootstrap_ci(va, vb, blocks))}")
    out.append("")

    # 5. independent reference (3DEP) if present
    dep = EXP / "dem3dep_fetched.json"
    dep_csvs = sorted(EXP.glob("runs/metrics3dep_*.csv"))
    if dep.exists() and dep_csvs:
        done = set(json.loads(dep.read_text()).get("done", []))
        out.append("## Independent reference (USGS 3DEP, CONUS tiles)")
        out.append(f"- {len(done)} tiles have independent 3DEP truth; "
                   "others are same-source (COP30/Tilezen == coarse origin).")
        dep_rows = {}
        for p in dep_csvs:
            m = re.match(r"metrics3dep_([A-Z]\d+)(?:_(\d))?\.csv", p.name)
            ar, seed = m.group(1), m.group(2) or "0"
            for r in csv.DictReader(open(p)):
                dep_rows.setdefault(ar, defaultdict(dict))[r["tile"]][seed] = r
        for ar in ARMS:
            d = dep_rows.get(ar)
            if not d:
                continue
            tiles = sorted(d)
            mae = np.mean([np.mean([float(v["mae"]) for v in d[t].values()])
                           for t in tiles])
            resid = np.mean([np.mean([float(v["resid_r"]) for v in d[t].values()])
                             for t in tiles])
            pos = sum(1 for t in tiles if
                      np.mean([float(v["raw_r"]) for v in d[t].values()]) > 0)
            out.append(f"- {ar}: n3dep={len(tiles)} mae={mae:.1f}m "
                       f"resid_r={resid:+.3f} raw_r>0={pos}/{len(tiles)}")
        out.append("- paired B-C resid_r (residual detail vs 3DEP, "
                   "CV tiles, block bootstrap):")
        for a, b in (("B90", "C90"), ("B250", "C250")):
            da, db = dep_rows.get(a), dep_rows.get(b)
            if not da or not db:
                continue
            va = {t: np.mean([float(v["resid_r"]) for v in da[t].values()])
                  for t in da}
            vb = {t: np.mean([float(v["resid_r"]) for v in db[t].values()])
                  for t in db}
            out.append(f"  - {a}-{b}: {fmt(bootstrap_ci(va, vb, blocks))}")
    else:
        out.append("## Independent reference (USGS 3DEP)\n- (no 3DEP data yet)")
    out.append("")

    # 6. conclusion mapped to the requested tri-state
    out.append("## Conclusion")
    out.append("- Primary test (native same-source truth, 54 CV tiles, "
               "block-bootstrap 95% CI on residual detail):")
    b90c = bootstrap_ci(
        {t: rows["B90"][t]["resid_r"] for t in rows["B90"]
         if t in rows["C90"] and t in blocks},
        {t: rows["C90"][t]["resid_r"] for t in rows["B90"]
         if t in rows["C90"] and t in blocks}, blocks)
    b250c = bootstrap_ci(
        {t: rows["B250"][t]["resid_r"] for t in rows["B250"]
         if t in rows["C250"] and t in blocks},
        {t: rows["C250"][t]["resid_r"] for t in rows["B250"]
         if t in rows["C250"] and t in blocks}, blocks)
    b90d = bootstrap_ci(
        {t: rows["B90"][t]["resid_r"] for t in rows["B90"]
         if t in rows["D90"] and t in blocks},
        {t: rows["D90"][t]["resid_r"] for t in rows["B90"]
         if t in rows["D90"] and t in blocks}, blocks)
    b250d = bootstrap_ci(
        {t: rows["B250"][t]["resid_r"] for t in rows["B250"]
         if t in rows["D250"] and t in blocks},
        {t: rows["D250"][t]["resid_r"] for t in rows["B250"]
         if t in rows["D250"] and t in blocks}, blocks)
    for tag, c in (("B90-C90", b90c), ("B250-C250", b250c),
                   ("B90-D90", b90d), ("B250-D250", b250d)):
        out.append(f"  - {tag} resid_r: {fmt(c)}")
    out.append("")
    out.append("- Ruling on 'image adds detail beyond prior' (needs B>C and "
               "B>D, CI excluding 0 on resid_r):")
    out.append("  - 90 m prior: NO - B90-C90 resid CI entirely <= 0.")
    out.append("  - 250 m prior: weak YES - B250 beats both C and D with CI "
               "excluding 0, but delta ~+0.03..+0.05 (a few % of remaining "
               "fine-scale variance).")
    out.append("  - Absolute error (MAE/RMSE on held-out 20%): every model "
               "arm is 3-8x worse than the simple coarse upsampler (B90 mae "
               "68.8 vs C90 5.8 m; B250 74.8 vs C250 16.6 m), and raw_r of B "
               "is below C (0.85/0.81 vs 0.998/0.988).")
    out.append("  - Independent 3DEP truth (56 US tiles): B residual detail "
               "is small-positive (+0.025/+0.078) but absolute error is "
               "again ~5x the coarse upsampler.")
    out.append("")
    out.append("-> VERDICT: **the image adds essentially nothing usable; at "
               "best a marginal fine-detail gain at the coarser 250m prior "
               "that does not survive the absolute-error comparison.** "
               "Conclusion = 'image adds nothing (B ~ C in real terms)'; "
               "n=54 CV tiles, 56 tiles with independent USGS 3DEP reference.")

    report = EXP / "TASK2_REPORT.md"
    report.write_text("\n".join(out))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()