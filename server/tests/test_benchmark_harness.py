"""Phase 1 benchmark-harness checks.

Invariants pinned here:
  - the manifest loader fails on a site spanning two splits and warns when
    cross-split tiles sit inside the guard distance; the shipped manifest
    validates clean;
  - the spatial block split is disjoint from its own guard band;
  - every arm is scored on ONE shared valid-pixel mask per tile;
  - an undefined (constant) raw correlation keeps its reason, never 0.0, and
    the tile stays in the failure counts;
  - the bicubic reference arm uses a real reprojection: a linear plane is
    reproduced exactly under cubic and blockily under nearest;
  - identical inputs produce identical output (deterministic caps/subsampling);
  - acceptance thresholds come from the config dict only.
"""
import json

import numpy as np
import pytest

from app.pipeline import evaluation as ev


def _ramp(h=120, w=120, a=10.0, b=0.5):
    y, x = np.mgrid[0:h, 0:w]
    return (a + b * x + 0.0 * y).astype("float32")


def _row(**kw):
    base = {
        "tile_id": "t1", "site_id": "site_a", "region": "site_a", "split": "test",
        "sensor": "s", "capture_date": "", "gsd": "30", "crs": "EPSG:4326",
        "reference_product": "p", "reference_type": "DSM", "reference_res_m": "30",
        "vertical_datum": "d", "stratum": "hilly", "reference_hr_path": "",
        "notes": "", "bounds_wgs84": "0.0,0.0,0.2,0.2",
    }
    base.update(kw)
    return base


# ---- manifest loader / validator ------------------------------------------- #
def test_manifest_loader_requires_columns_and_sites(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("tile_id,split\nx,test\n", encoding="utf-8")
    with pytest.raises(ev.ManifestError):
        ev.load_manifest(p)


def test_manifest_validator_flags_site_leak():
    rows = [_row(tile_id="t1"),
            _row(tile_id="t2", split="train")]
    v = ev.validate_manifest(rows, guard_km=10.0)
    assert any("multiple splits" in e for e in v["errors"])


def test_manifest_validator_guard_band_warning():
    rows = [
        _row(tile_id="t1", bounds_wgs84="0.0,0.0,0.2,0.2"),
        _row(tile_id="t2", site_id="b", region="b", split="train",
             bounds_wgs84="0.21,0.0,0.41,0.2"),
    ]
    v = ev.validate_manifest(rows, guard_km=10.0)
    assert v["errors"] == []
    assert any("km apart" in w for w in v["warnings"])
    # tightening the configurable guard to below the actual gap clears it
    assert ev.validate_manifest(rows, guard_km=1.0)["warnings"] == []


def test_shipped_manifest_validates():
    path = ev._REPO_ROOT / "server" / "data" / "manifest.csv"
    rows, validation = ev.load_manifest(path)
    assert len(rows) == 54
    assert {r["split"] for r in rows} == {"train", "test"}
    assert validation["warnings"] == []
    assert {r["site_id"] for r in rows if r["split"] == "test"} == {
        "appalachians_smokies", "western_ghats", "n_arizona", "mojave", "new_mexico"}


# ---- spatial block split ---------------------------------------------------- #
def test_spatial_block_split_disjoint_with_guard():
    from scipy import ndimage

    calib, evalm = ev.spatial_block_split((97, 130), block=16, guard=3)
    assert not np.any(calib & evalm)
    blocked = ndimage.binary_dilation(
        calib, structure=np.ones((2 * 3 + 1, 2 * 3 + 1), dtype=bool))
    assert not np.any(blocked & evalm)
    assert calib.mean() > 0.4
    # the guard band costs a large share at these small test blocks (16 px
    # blocks, 3 px each side); real runs use block=180/guard=8
    assert evalm.mean() > 0.15


# ---- shared masks, honesty, determinism ------------------------------------- #
def test_arms_share_one_valid_mask():
    dem = _ramp()
    p1 = _ramp(a=5.0, b=0.2)
    p2 = _ramp(a=1.0, b=0.1)
    # holes inside EVAL blocks (block=20: block (1,0) rows 20..39 / cols 0..19
    # has parity 1 = eval; well clear of the 2 px guard band)
    p1[24:36, 4:16] = np.nan
    p2[26:38, 2:14] = np.nan
    rec = ev.evaluate_tile_arrays(dem, {"one": p1, "two": p2},
                                  block=20, guard=2, seed=1)
    _, evalm = ev.spatial_block_split(dem.shape, block=20, guard=2)
    expected = int(np.count_nonzero(
        evalm & np.isfinite(dem) & np.isfinite(p1) & np.isfinite(p2)))
    assert rec["n_eval_pixels"] == expected
    assert expected < int(np.count_nonzero(evalm))  # the holes are inside eval
    for info in rec["backends"].values():
        assert info["raw_r"] is not None  # both arms scored on the same pixels


def test_undefined_raw_correlation_keeps_reason_not_zero():
    dem = _ramp()
    flat = np.full_like(dem, 7.0)
    rec = ev.evaluate_tile_arrays(dem, {"flat": flat}, block=20, guard=2, seed=1)
    info = rec["backends"]["flat"]
    assert info["raw_r"] is None
    assert info["raw_r_reason"] and "zero variance" in info["raw_r_reason"]
    assert info["status"] == "failed"  # degenerate calibration, still counted
    assert info["cal_rmse"] is None


def test_failure_stats_count_failed_and_valid_tiles():
    dem = _ramp()
    good = (5.0 + 0.2 * (dem - float(dem.min()))).astype("float32")
    inverted = (35.0 - 0.2 * (dem - float(dem.min()))).astype("float32")
    r_ok = ev.evaluate_tile_arrays(dem, {"bk": good}, block=20, guard=2, seed=1)
    r_inv = ev.evaluate_tile_arrays(dem, {"bk": inverted}, block=20, guard=2, seed=1)
    r_bad = ev.evaluate_tile_arrays(dem, {"bk": np.full_like(dem, 3.0)},
                                    block=20, guard=2, seed=1)
    assert r_inv["backends"]["bk"]["estimate_status"] == "low_confidence"
    for rec, tile_id in ((r_ok, "t1"), (r_inv, "t2"), (r_bad, "t3")):
        rec.update({"tile_id": tile_id, "site_id": "s", "stratum": "hilly",
                    "split": "test"})
    agg = ev.aggregate_records([r_ok, r_inv, r_bad], bootstrap_iters=50, seed=1)
    failures = agg["arms"]["bk"]["failures"]
    assert failures["valid"] == 1
    assert failures["low_confidence"] == 1
    assert failures["failed"] == 1
    assert failures["failure_rate"] == pytest.approx(1.0 / 3.0)
    assert failures["low_confidence_rate"] == pytest.approx(1.0 / 3.0)


def test_bicubic_reference_uses_real_reprojection():
    from rasterio.transform import from_origin

    dem = _ramp()
    rec = ev.evaluate_tile_arrays(dem, {}, block=20, guard=2, seed=1)
    # reference and evaluation share one grid: identity reprojection
    assert rec["reference"]["nearest"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert rec["reference"]["bicubic"]["rmse"] == pytest.approx(0.0, abs=1e-6)

    # genuinely coarser source: cubic reproduces a linear plane exactly,
    # nearest does not - i.e. the resampling method is actually applied
    coarse = (10.0 + np.arange(60, dtype="float32")[:, None]
              + 0.0 * np.arange(60, dtype="float32")[None, :])
    t_coarse = from_origin(0.0, 1.2, 0.02, 0.02)
    t_fine = from_origin(0.0, 1.2, 0.01, 0.01)
    georef = (t_coarse, "EPSG:4326", t_fine, "EPSG:4326")
    cubic = ev._resample_reference(coarse, (120, 120), georef=georef,
                                   resampling="cubic")
    nearest = ev._resample_reference(coarse, (120, 120), georef=georef,
                                     resampling="nearest")
    expected = np.broadcast_to(
        (9.75 + 0.5 * np.arange(120, dtype="float64"))[:, None], (120, 120))
    interior = slice(10, 110)
    assert np.max(np.abs(cubic[interior, interior] - expected[interior, interior])) < 0.01
    assert np.max(np.abs(nearest[interior, interior] - expected[interior, interior])) > 0.1


# ---- baselines: prior degradation + constant mean ---------------------------- #
def test_prior_degradation_baseline_degrades_reference():
    # a linear plane survives block-average + cubic (plane-preserving): the
    # baseline must not invent error where a coarse prior is lossless
    plane = _ramp()
    rec = ev.evaluate_tile_arrays(plane, {}, gsd_m=10.0, prior_res_m=(50.0, 250.0),
                                  block=20, guard=2, seed=1)
    assert rec["baselines"]["prior_50m"]["status"] == "ok"
    assert rec["baselines"]["prior_50m"]["rmse"] < 0.2

    # a high-frequency surface is destroyed by the same degradation
    y, x = np.mgrid[0:120, 0:120]
    wavy = (100.0 + 50.0 * np.sin(2.0 * np.pi * x / 8.0)).astype("float32")
    rec2 = ev.evaluate_tile_arrays(wavy, {}, gsd_m=10.0, prior_res_m=(90.0,),
                                   block=20, guard=2, seed=1)
    assert rec2["baselines"]["prior_90m"]["rmse"] > 0.5 * rec2["reference_std"]

    # nodata in the reference neither blanks the baseline nor leaks into stats
    holed = _ramp().copy()
    holed[0:30, 0:30] = np.nan
    rec3 = ev.evaluate_tile_arrays(holed, {}, gsd_m=10.0, prior_res_m=(50.0,),
                                   block=20, guard=2, seed=1)
    assert rec3["baselines"]["prior_50m"]["status"] == "ok"
    assert rec3["baselines"]["prior_50m"]["rmse"] is not None

    # unavailable without a usable gsd / when not coarser than the native grid
    rec4 = ev.evaluate_tile_arrays(plane, {}, prior_res_m=(50.0,),
                                   block=20, guard=2, seed=1)
    assert rec4["baselines"]["prior_50m"]["status"] == "unavailable"
    rec5 = ev.evaluate_tile_arrays(plane, {}, gsd_m=10.0, prior_res_m=(5.0,),
                                   block=20, guard=2, seed=1)
    assert rec5["baselines"]["prior_5m"]["status"] == "unavailable"


def test_constant_mean_baseline_and_reference_std_use_calibration_split():
    dem = _ramp()
    rec = ev.evaluate_tile_arrays(dem, {}, block=20, guard=2, seed=1)
    calib_m, eval_m = ev.spatial_block_split(dem.shape, block=20, guard=2)
    calib_px = calib_m & np.isfinite(dem)
    eval_px = eval_m & np.isfinite(dem)

    b = rec["baselines"]["const_mean"]
    expected = float(np.mean(dem[calib_px].astype("float64")))
    assert b["status"] == "ok"
    assert b["value"] == pytest.approx(expected)
    # scored against the eval split only: no calibration-pixel leakage
    assert b["rmse"] == pytest.approx(
        float(np.sqrt(np.mean((expected - dem[eval_px].astype("float64")) ** 2))))
    assert rec["reference_std"] == pytest.approx(
        float(np.std(dem[eval_px].astype("float64"))))


def test_baseline_improvement_and_skill_aggregate():
    dem = _ramp()
    good = (5.0 + 0.2 * (dem - float(dem.min()))).astype("float32")
    rec = ev.evaluate_tile_arrays(dem, {"bk": good}, gsd_m=10.0,
                                  prior_res_m=(50.0,), block=20, guard=2, seed=1)
    rec.update({"tile_id": "t1", "site_id": "s", "stratum": "hilly",
                "split": "test"})
    agg = ev.aggregate_records([rec], bootstrap_iters=10, seed=1)
    arm = agg["arms"]["bk"]
    assert agg["baseline_prior"] == "prior_50m"
    prior_rmse = agg["baselines"]["prior_50m"]["rmse"]
    assert arm["improvement_vs_baseline"] == pytest.approx(
        (prior_rmse - arm["cal_rmse"]) / prior_rmse)
    assert arm["skill_vs_const"] == pytest.approx(
        1.0 - arm["cal_rmse"] / agg["baselines"]["const_mean"]["rmse"])
    assert arm["contaminated"] is False


def test_contaminated_arm_excluded_from_primary_choice():
    dem = _ramp()
    good = (5.0 + 0.2 * (dem - float(dem.min()))).astype("float32")
    noisy = (good + np.random.default_rng(3).normal(0, 8.0, dem.shape)).astype("float32")
    rec = ev.evaluate_tile_arrays(dem, {"finetuned": good, "da2": noisy},
                                  block=20, guard=2, seed=1)
    rec.update({"tile_id": "t1", "site_id": "s", "stratum": "hilly",
                "split": "test"})
    agg = ev.aggregate_records([rec], bootstrap_iters=10, seed=1)
    assert agg["arms"]["finetuned"]["contaminated"] is True
    assert agg["arms"]["da2"]["contaminated"] is False
    # finetuned has the lower RMSE here, yet da2 is primary: contamination
    # disqualifies it from the choice, not just from the label
    assert agg["arms"]["finetuned"]["cal_rmse"] < agg["arms"]["da2"]["cal_rmse"]
    assert agg["primary_arm"] == "da2"


def test_benchmark_deterministic_for_same_inputs():
    dem = _ramp()
    pred = _ramp(a=2.0, b=0.3)
    pred[30:40, 30:40] = np.nan
    r1 = ev.evaluate_tile_arrays(dem, {"bk": pred}, block=20, guard=2,
                                 calib_cap=500, seed=7)
    r2 = ev.evaluate_tile_arrays(dem, {"bk": pred}, block=20, guard=2,
                                 calib_cap=500, seed=7)
    assert (json.dumps(r1, sort_keys=True, default=str)
            == json.dumps(r2, sort_keys=True, default=str))


# ---- acceptance thresholds --------------------------------------------------- #
def _stub_aggregate():
    return {
        "primary_arm": "bk",
        "arms": {"bk": {"cal_rmse": 1.0, "raw_r": 0.5,
                        "improvement_vs_baseline": None,
                        "failures": {"failure_rate": 0.0}}},
        "per_stratum": {"hilly": {"arms": {
            "bk": {"improvement_vs_baseline": None}}}},
    }


def test_acceptance_reads_thresholds_only_from_config():
    cfg = {"minimum_improvement": 0.05, "per_stratum_improvement": {"hilly": 0.0},
           "max_failure_rate": 0.10, "min_raw_correlation": 0.20, "min_test_sites": 3}
    agg = _stub_aggregate()
    # no prior-degradation baseline (improvement None) -> not measurable, not pass
    assert ev.evaluate_acceptance(cfg, agg, n_test_sites=3)["verdict"] == "not_measurable"
    # the same aggregate FAILS once the shipped failure threshold is exceeded
    agg["arms"]["bk"]["failures"]["failure_rate"] = 0.5
    assert ev.evaluate_acceptance(cfg, agg, n_test_sites=3)["verdict"] == "fail"
    # and the site-count floor is read from the config too
    assert ev.evaluate_acceptance(cfg, _stub_aggregate(), n_test_sites=2)["verdict"] == "fail"
