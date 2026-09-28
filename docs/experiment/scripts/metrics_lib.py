"""Unified scorer for the RGB-vs-coarse-prior experiment (arms B/C/D).

Per tile, on a deterministic 80/20 pixel split:
    affine   model relative predictions (0..1) -> meters via polyfit on 80%
    mae/rmse vs truth (test 20%)
    raw_r     corr(pred_elev, truth)          -- affine-invariant
    resid_r   corr(pred_elev-coarse_up, truth-coarse_up)
              = detail-skill above the coarse prior; == 0 for a pure upsampler.
C baseline: pred == coarse_up (already meters), no affine.

Self-check (python metrics_lib.py): resid_r(C)==0, raw_r(C)>0.8, determinism,
affine invariance of raw_r.
"""
import hashlib
from pathlib import Path

import numpy as np
import rasterio
from scipy import stats

REPO = Path(r"C:\Users\harsh\Downloads\files\DepthWizard")
HELDOUT = Path(r"C:\Users\harsh\Downloads\files\terrain_stratum_test\data")
GEOTEST = Path(r"C:\Users\harsh\Downloads\files\geo_test_data")
EXP = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode\exp")


def dem_path(tile_id):
    if "/" in tile_id:
        fam, name = tile_id.split("/", 1)
        return REPO / "training_data" / fam / f"{name}_dem.tif"
    if tile_id in ("colorado_original", "colorado_north", "colorado_west"):
        return GEOTEST / f"{tile_id}_dem.tif"
    return HELDOUT / f"{tile_id}_dsm.tif"


def load_truth(tile_id):
    with rasterio.open(dem_path(tile_id)) as s:
        arr = s.read(1).astype("float64")
        nod = s.nodata
    if nod is not None:
        arr = np.where(arr == nod, np.nan, arr)
    return arr


def coarse_up(tile_id, res):
    return np.load(EXP / "coarse_elev" / f"{tile_id.replace('/', '_')}_{res}.npy")


def _split_seed(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2**32)


def _pearson(a, b):
    if len(a) < 3:
        return float("nan")
    return float(stats.pearsonr(a, b)[0])


def score(pred, truth, cu, key, affine=True, frac_train=0.8):
    """pred: meters (or model-relative-depth + affine=True). Returns dict or
    None if the tile has too few valid pixels."""
    valid = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(cu)
    x = pred[valid].ravel().astype("float64")
    y = truth[valid].ravel().astype("float64")
    c = cu[valid].ravel().astype("float64")
    if len(x) < 512:
        return None
    rng = np.random.RandomState(_split_seed(key))
    perm = rng.permutation(len(x))
    n_tr = max(64, int(round(len(x) * frac_train)))
    tr, te = perm[:n_tr], perm[n_tr:]
    if len(te) < 64:
        return None
    if affine:
        a, b = np.polyfit(x[tr], y[tr], 1)
        pe = a * x[te] + b
    else:
        pe = x[te]
    yt, ct = y[te], c[te]
    d_pred, d_truth = pe - ct, yt - ct
    if np.allclose(d_pred, 0.0):
        resid_r = 0.0
    else:
        resid_r = _pearson(d_pred, d_truth)
    return dict(
        mae=float(np.mean(np.abs(pe - yt))),
        rmse=float(np.sqrt(np.mean((pe - yt) ** 2))),
        raw_r=_pearson(pe, yt),
        resid_r=resid_r,
        raw_pos=int(_pearson(pe, yt) > 0),
        n_valid=int(len(x)),
    )


def score_tile_c(tile_id, res):
    t = load_truth(tile_id)
    return score(coarse_up(tile_id, res), t, coarse_up(tile_id, res), tile_id,
                 affine=False)


def score_tile_model(tile_id, res, pred):
    t = load_truth(tile_id)
    return score(pred, t, coarse_up(tile_id, res), tile_id, affine=True)


if __name__ == "__main__":
    for tid, res in (("hilly/hilly_05", "90"), ("sparse/sparse_09", "250")):
        m = score_tile_c(tid, res)
        assert m and abs(m["resid_r"]) < 1e-12, (tid, m)
        assert m["raw_r"] > 0.8, (tid, m)
        assert isinstance(m["raw_pos"], int)
        m2 = score_tile_c(tid, res)
        assert m == m2, "determinism"
        print(f"C{res} {tid}: mae={m['mae']:.1f}m rmse={m['rmse']:.1f}m "
              f"raw_r={m['raw_r']:.3f} resid_r={m['resid_r']}")

    t = load_truth("hilly/hilly_05")
    cu = coarse_up("hilly/hilly_05", "90")
    fake = (t - t.min()) / (t.max() - t.min())
    ra = score(fake, t, cu, "fake", affine=True)["raw_r"]
    rb = score(fake, t, cu, "fake", affine=False)["raw_r"]
    assert abs(ra - rb) < 1e-9, (ra, rb)
    print(f"affine invariance raw_r: {ra:.6f} == {rb:.6f}")
    print("self-check OK")