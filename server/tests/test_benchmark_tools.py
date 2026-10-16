"""Focused tests for tools/benchmark_tiles.py (synthetic fixtures, no model).

Covers: region assignment from actual bounds, the verdict logic mirrored from
tools/score_my_data.py, the historical protocol end-to-end with an injected
fake model, the geographic protocol's dev-only frozen calibration, the
local-calibration spatial buffer, unique run directories, and the reference
parser for the historical artifact.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = REPO_ROOT / "tools" / "benchmark_tiles.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("benchmark_tiles", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_tiles"] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load_tool()


def _fake_infer(rgb):
    h, w = rgb.shape[:2]
    return np.repeat(np.arange(w, dtype="float64")[None, :] / (w - 1), h, axis=0)


def _write_pair(root: Path, stratum: str, tile_id: str, dem_func,
                west=0.0, north=1.0, size=128, cell=0.002, crs="EPSG:4326"):
    import rasterio
    from rasterio.transform import from_origin

    d = root / stratum
    d.mkdir(parents=True, exist_ok=True)
    tf = from_origin(west, north, cell, cell)
    yy, xx = np.mgrid[0:size, 0:size]
    rdsm = xx / (size - 1)
    dem = dem_func(rdsm).astype("float32")
    with rasterio.open(d / f"{tile_id}_dem.tif", "w", driver="GTiff", height=size,
                       width=size, count=1, dtype="float32", crs=crs, transform=tf) as dst:
        dst.write(dem, 1)
    rgb = np.stack([(xx / (size - 1) * 255).astype("uint8"),
                    np.full((size, size), 128, np.uint8),
                    (yy / (size - 1) * 255).astype("uint8")])
    with rasterio.open(d / f"{tile_id}_rgb.tif", "w", driver="GTiff", height=size,
                       width=size, count=3, dtype="uint8", crs=crs, transform=tf) as dst:
        dst.write(rgb)
    return d


def _run_args(**over):
    base = dict(protocol="historical", backend="auto", model_path=None,
                root=None, run_root=None, manifest=None, tiles=None, stratum=None,
                limit=None, ckpt=None, seed=42, calibration_pixel_cap=500_000,
                local_buffer_deg=0.2, save_predictions=True, save_comparisons=True,
                reference=None, reuse_predictions=None, allow_unverified_reuse=False,
                compare_tol=1e-3)
    base.update(over)
    return argparse.Namespace(**base)


def _counting_infer():
    calls = []

    def fn(rgb):
        calls.append(1)
        return _fake_infer(rgb)

    return fn, calls


def _read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
def test_region_assignment_from_bounds():
    assert bt.assign_region((-106.7, 40.1, -106.5, 40.3)) == "colorado_front_range"
    assert bt.assign_region((-107.1, 39.3, -106.9, 39.5)) == "colorado_front_range"
    assert bt.assign_region((74.0, 15.0, 74.2, 15.2)) == "western_ghats"
    assert bt.assign_region((-117.2, 36.7, -117.0, 36.9)) == "mojave"
    assert bt.assign_region((-106.3, 32.5, -106.1, 32.7)) == "new_mexico"
    assert bt.assign_region((-83.6, 35.5, -83.4, 35.7)) == "appalachians_smokies"
    assert bt.assign_region((0.0, 0.0, 0.2, 0.2)) is None


def test_verdict_mirrors_score_my_data():
    v = bt.verdict(-0.4, +0.4)
    assert (v["flag"], v["polarity"], v["stability"]) == ("SIGN-INVERTED", "inverted", "flipped")
    v = bt.verdict(+0.1, +0.1)
    assert (v["flag"], v["polarity"], v["stability"]) == ("weak-raw", "no-signal", "stable")
    v = bt.verdict(+0.5, -0.2)
    assert (v["flag"], v["polarity"], v["stability"]) == ("REVERSED", "positive", "flipped")
    v = bt.verdict(+0.3, +0.3)
    assert (v["flag"], v["polarity"], v["stability"]) == ("ok", "positive", "stable")
    v = bt.verdict(None, None)
    assert (v["flag"], v["polarity"], v["stability"]) == ("ok", "n/a", "n/a")
    v = bt.verdict(-0.5, None)
    assert (v["flag"], v["polarity"], v["stability"]) == ("ok", "inverted", "n/a")


def test_historical_protocol_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _write_pair(root, "hilly", "good_00", lambda d: 1000.0 * d + 50.0)
    _write_pair(root, "hilly", "bad_00", lambda d: 1000.0 * (1.0 - d) + 50.0)
    monkeypatch.setattr(bt, "_default_infer", _fake_infer)
    run_root = tmp_path / "runs"
    args = _run_args(root=str(root), run_root=str(run_root), ckpt=str(tmp_path / "none.pt"))
    assert bt.cmd_run(args) == 0

    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    rows = {r["tile_id"]: r for r in _read_rows(run_dirs[0] / "per_tile.csv")}
    good, bad = rows["good_00"], rows["bad_00"]
    assert float(good["raw_corr"]) > 0.999
    assert float(good["rmse"]) < 0.01
    assert good["flag"] == "ok"
    assert float(bad["raw_corr"]) < -0.999
    assert bad["flag"] == "SIGN-INVERTED"
    assert bad["polarity"] == "inverted"
    assert (run_dirs[0] / "summary.json").exists()
    assert (run_dirs[0] / "per_tile" / "good_00.json").exists()
    assert (run_dirs[0] / "predictions" / "good_00_relative.npy").exists()
    assert (run_dirs[0] / "comparisons" / "good_00.png").exists()

    summary = json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["tiles"]["succeeded"] == 2
    assert summary["flags"]["SIGN-INVERTED"] == 1

    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"))) == 0
    assert len(list(run_root.iterdir())) == 2


def test_geographic_frozen_calibration_fits_dev_only(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _write_pair(root, "d", "dev_a", lambda d: 1000.0 * d + 50.0, west=0.0)
    _write_pair(root, "d", "dev_b", lambda d: 1000.0 * d + 50.0, west=0.4)
    _write_pair(root, "t", "test_a", lambda d: 7.0 * d + 1500.0, west=10.0)
    _write_pair(root, "t", "test_b", lambda d: 7.0 * d + 1500.0, west=10.4)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "tile_id,region,protocol_b_split\n"
        "dev_a,dev_region,dev\ndev_b,dev_region,dev\n"
        "test_a,test_region,test\ntest_b,test_region,test\n", encoding="utf-8")
    monkeypatch.setattr(bt, "_default_infer", _fake_infer)
    run_root = tmp_path / "runs"
    args = _run_args(protocol="geographic", root=str(root), run_root=str(run_root),
                     manifest=str(manifest), ckpt=str(tmp_path / "none.pt"))
    assert bt.cmd_run(args) == 0
    summary = json.loads(next(run_root.iterdir()).joinpath("summary.json").read_text(encoding="utf-8"))
    cal = summary["calibration"]
    assert cal["dev_tiles"] == ["dev_a", "dev_b"]
    assert cal["test_tiles"] == ["test_a", "test_b"]
    assert abs(cal["scale"] - 1000.0) < 0.1
    assert abs(cal["offset"] - 50.0) < 0.1
    groups = summary["groups"]
    assert set(groups) == {"test_region"}
    assert groups["test_region"]["n_tiles"] == 2
    rows = {r["tile_id"]: r for r in _read_rows(next(run_root.iterdir()) / "per_tile.csv")}
    # Frozen scale 1000 applied to test tiles whose true relation is 7 => huge RMSE.
    assert float(rows["test_a"]["rmse"]) > 100.0
    assert rows["test_a"]["baseline_rmse"] is not None


def test_local_calibration_spatial_buffer(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for i, west in enumerate((0.0, 0.2, 0.4, 0.6)):
        _write_pair(root, "r", f"t{i}", lambda d: 100.0 * d + 10.0, west=west,
                    cell=0.2 / 128)
    monkeypatch.setattr(bt, "_default_infer", _fake_infer)
    run_root = tmp_path / "runs"
    args = _run_args(protocol="local_calibration", root=str(root), run_root=str(run_root),
                     ckpt=str(tmp_path / "none.pt"))
    assert bt.cmd_run(args) == 0
    summary = json.loads(next(run_root.iterdir()).joinpath("summary.json").read_text(encoding="utf-8"))
    region = summary["regions"]["unassigned"]
    assert region["calibration_block"] == ["t0", "t1"]
    assert region["test_block"] == ["t3"]
    assert region["buffered_out"] == ["t2"]


def test_reuse_predictions_skips_model(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _write_pair(root, "hilly", "good_00", lambda d: 1000.0 * d + 50.0)
    monkeypatch.setattr(bt, "_default_infer", _fake_infer)
    run_root = tmp_path / "runs"
    args = _run_args(root=str(root), run_root=str(run_root), ckpt=str(tmp_path / "none.pt"))
    assert bt.cmd_run(args) == 0
    first = next(run_root.iterdir())
    first_rows = _read_rows(first / "per_tile.csv")
    # predictions always carry an identity sidecar next to the .npy
    sidecar = first / "predictions" / "good_00_relative.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["provenance"] == "inference"
    assert meta["prediction_sha256"]

    def _boom(rgb):
        raise AssertionError("inference must not run when predictions are reused")

    monkeypatch.setattr(bt, "_default_infer", _boom)
    args2 = _run_args(root=str(root), run_root=str(run_root), ckpt=str(tmp_path / "none.pt"),
                      reuse_predictions=str(first / "predictions"))
    assert bt.cmd_run(args2) == 0
    second = [d for d in run_root.iterdir() if d != first][0]
    second_rows = _read_rows(second / "per_tile.csv")
    # cached predictions are the float32 .npy artifacts: equal up to float32 rounding
    assert float(second_rows[0]["rmse"]) == pytest.approx(float(first_rows[0]["rmse"]), abs=1e-3)
    detail = json.loads((second / "per_tile" / "good_00.json").read_text(encoding="utf-8"))
    assert detail["prediction_reused"] is True
    assert detail["prediction_provenance"] == "reuse_verified"
    run_meta = json.loads((second / "run.json").read_text(encoding="utf-8"))
    assert run_meta["reuse"]["reused"] == ["good_00"]
    assert run_meta["reuse"]["regenerated"] == {}


def test_reuse_rejects_changed_tile_inputs(tmp_path, monkeypatch):
    import rasterio

    root = tmp_path / "data"
    _write_pair(root, "hilly", "good_00", lambda d: 1000.0 * d + 50.0)
    monkeypatch.setattr(bt, "_default_infer", _fake_infer)
    run_root = tmp_path / "runs"
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"))) == 0
    first = next(run_root.iterdir())

    rgb_path = root / "hilly" / "good_00_rgb.tif"
    with rasterio.open(rgb_path, "r+") as dst:
        dst.write(dst.read(2) + 1, 2)  # same shape, different bytes

    fn, calls = _counting_infer()
    monkeypatch.setattr(bt, "_default_infer", fn)
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"),
                                reuse_predictions=str(first / "predictions"))) == 0
    second = [d for d in run_root.iterdir() if d != first][0]
    detail = json.loads((second / "per_tile" / "good_00.json").read_text(encoding="utf-8"))
    assert detail["prediction_reused"] is False
    assert "changed" in detail["reuse_note"]
    assert calls  # stale cache must trigger a re-run, never a silent reuse
    run_meta = json.loads((second / "run.json").read_text(encoding="utf-8"))
    assert "good_00" in run_meta["reuse"]["regenerated"]


def test_reuse_rejects_changed_inference_code_and_model(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _write_pair(root, "hilly", "good_00", lambda d: 1000.0 * d + 50.0)
    monkeypatch.setattr(bt, "_default_infer", _fake_infer)
    run_root = tmp_path / "runs"
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"))) == 0
    first = next(run_root.iterdir())

    fn, calls = _counting_infer()
    monkeypatch.setattr(bt, "_default_infer", fn)
    orig_fingerprint = bt._inference_fingerprint
    monkeypatch.setattr(bt, "_inference_fingerprint", lambda: "changed-code")
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"),
                                reuse_predictions=str(first / "predictions"))) == 0
    second = [d for d in run_root.iterdir() if d != first][0]
    detail = json.loads((second / "per_tile" / "good_00.json").read_text(encoding="utf-8"))
    assert detail["prediction_reused"] is False
    assert "inference code changed" in detail["reuse_note"]
    assert calls

    orig_model_info = bt._model_info

    def fake_model_info(backend):
        info = orig_model_info(backend)
        info["file_sha256"] = {**info["file_sha256"], "saved_model.pb": "0" * 64}
        return info

    monkeypatch.setattr(bt, "_inference_fingerprint", orig_fingerprint)
    monkeypatch.setattr(bt, "_model_info", fake_model_info)
    fn2, calls2 = _counting_infer()
    monkeypatch.setattr(bt, "_default_infer", fn2)
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"),
                                reuse_predictions=str(first / "predictions"))) == 0
    third = [d for d in run_root.iterdir() if d not in (first, second)][0]
    detail3 = json.loads((third / "per_tile" / "good_00.json").read_text(encoding="utf-8"))
    assert detail3["prediction_reused"] is False
    assert "model changed" in detail3["reuse_note"]
    assert calls2


def test_legacy_cache_needs_explicit_flag(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _write_pair(root, "hilly", "good_00", lambda d: 1000.0 * d + 50.0)
    monkeypatch.setattr(bt, "_default_infer", _fake_infer)
    run_root = tmp_path / "runs"
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"))) == 0
    first = next(run_root.iterdir())
    for sidecar in (first / "predictions").glob("*.json"):
        sidecar.unlink()  # pretend the cache predates identity sidecars

    fn, calls = _counting_infer()
    monkeypatch.setattr(bt, "_default_infer", fn)
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"),
                                reuse_predictions=str(first / "predictions"))) == 0
    second = [d for d in run_root.iterdir() if d != first][0]
    detail = json.loads((second / "per_tile" / "good_00.json").read_text(encoding="utf-8"))
    assert detail["prediction_reused"] is False
    assert "no cache metadata" in detail["reuse_note"]
    assert calls

    def _boom(rgb):
        raise AssertionError("inference must not run when predictions are reused")

    monkeypatch.setattr(bt, "_default_infer", _boom)
    assert bt.cmd_run(_run_args(root=str(root), run_root=str(run_root),
                                ckpt=str(tmp_path / "none.pt"),
                                reuse_predictions=str(first / "predictions"),
                                allow_unverified_reuse=True)) == 0
    third = [d for d in run_root.iterdir() if d not in (first, second)][0]
    detail3 = json.loads((third / "per_tile" / "good_00.json").read_text(encoding="utf-8"))
    assert detail3["prediction_reused"] is True
    assert detail3["prediction_provenance"] == "legacy_unverified"
    run_meta = json.loads((third / "run.json").read_text(encoding="utf-8"))
    assert run_meta["reuse"]["reused_unverified"] == ["good_00"]


def test_parse_reference_utf16(tmp_path):
    text = (
        "Picked up 2 pair(s).\n"
        "    hilly_00_rgb.tif  +  hilly_00_dem.tif\n"
        "    hilly_01_rgb.tif  +  hilly_01_dem.tif\n"
        "[1/2] hilly\n"
        "========================================================================\n"
        "tile                      raw_corr  cal_corr      rmse       mae    held_out   flag\n"
        "------------------------------------------------------------------------\n"
        "hilly                       -0.500    +0.500    100.00     80.00      103680   SIGN-INVERTED\n"
        "    (calibration flipped a clearly negative raw correlation into a positive calibrated one)\n"
        "hilly                       +0.100    +0.100     50.00     40.00      103680   weak-raw\n"
        "========================================================================\n"
    )
    path = tmp_path / "ref.txt"
    path.write_bytes(text.encode("utf-16"))
    rows = bt._parse_reference(path)
    assert [r["tile_id"] for r in rows] == ["hilly_00", "hilly_01"]
    assert rows[0]["raw_corr"] == pytest.approx(-0.5)
    assert rows[0]["flag"] == "SIGN-INVERTED"
    assert rows[1]["mae"] == pytest.approx(40.0)
