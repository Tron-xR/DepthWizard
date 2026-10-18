"""RS height backends + ndsm/dsm evaluation modes.

Covers the remote-sensing height additions:
- rs3dada / altimap v1/v2 are strictly opt-in, single-select, and can never
  silently mix with a DA3 backend (depth dispatch unchanged when none set)
- metric prediction path: predict_height normalisation/windowing/blending/
  clamping with a stub model, and infer_ndsm_metres guards relative backends
- rs_checkpoint_provenance pins bytes and carries the licence note
- evaluate_metric_arrays: predict-zero/const_mean baselines, raw vs calibrated
  headline, failed tiles stay in counts
- aggregate_metric_records: skill vs predict-zero, primary-arm choice
- run_height_benchmark end-to-end on synthetic GeoTIFFs + npy predictions
  (ndsm mode and dsm mode with oracle/prior baselines)
- tools/benchmark_tiles registers rs backends (_force_backend/_model_info)
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import config
from app.pipeline import depth, evaluation

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = REPO_ROOT / "tools" / "benchmark_tiles.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("benchmark_tiles", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_tiles"] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load_tool()


def _clear_all_backends(monkeypatch):
    monkeypatch.setattr(config, "FINETUNED_MODEL_PATH", "")
    monkeypatch.setattr(config, "IMELE_MODEL_PATH", "")
    monkeypatch.setattr(config, "TF_MODEL_PATH", "")
    monkeypatch.setattr(depth, "_model_cache", {})
    for _slug, attr, _repo, _fn, _note in depth.RS_BACKENDS:
        monkeypatch.setattr(config, attr, "")
    for _slug, attr, _default in depth.DA3_BACKENDS:
        monkeypatch.setattr(config, attr, "")


# ---------------------------------------------------------------------------
# opt-in selection / default dispatch
# ---------------------------------------------------------------------------
def test_selected_rs_off_by_default_and_single_select(monkeypatch):
    _clear_all_backends(monkeypatch)
    assert depth._selected_rs() is None
    assert depth.active_output_kind() == "relative"  # default dispatch unchanged

    monkeypatch.setattr(config, "RS3DADA_MODEL", "JTRNEO/RS3DAda")
    sel = depth._selected_rs()
    assert sel[0] == "rs3dada" and sel[1] == "JTRNEO/RS3DAda"
    assert sel[2] == "RS3DAda_vitl_DPT_height.pth"
    assert depth.active_output_kind() == "ndsm_metres"
    assert depth.backend_slug() == "rs3dada"
    assert depth.model_identifier() == "JTRNEO/RS3DAda"

    monkeypatch.setattr(config, "ALTIMAP_V1_MODEL", "Dilavesh/altimap-height")
    with pytest.raises(depth.DepthModelError):
        depth._selected_rs()


def test_altimap_v2_filename_and_license_flag(monkeypatch):
    _clear_all_backends(monkeypatch)
    monkeypatch.setattr(config, "ALTIMAP_V2_MODEL", "Dilavesh/altimap-height")
    sel = depth._selected_rs()
    assert sel[2] == "v2/best.pth"  # file path is the version pin
    assert "NON-COMMERCIAL" in sel[3]


def test_rs_and_da3_configured_together_is_an_error(monkeypatch):
    _clear_all_backends(monkeypatch)
    monkeypatch.setattr(config, "RS3DADA_MODEL", "JTRNEO/RS3DAda")
    monkeypatch.setattr(config, "DA3_BASE_MODEL", "depth-anything/DA3-BASE")
    with pytest.raises(depth.DepthModelError):
        depth._load_model()


def test_finetuned_outranks_rs_in_output_kind(monkeypatch):
    _clear_all_backends(monkeypatch)
    monkeypatch.setattr(config, "RS3DADA_MODEL", "JTRNEO/RS3DAda")
    monkeypatch.setattr(config, "FINETUNED_MODEL_PATH", "C:/ckpt.pth")
    assert depth.active_output_kind() == "relative"


# ---------------------------------------------------------------------------
# metric prediction path
# ---------------------------------------------------------------------------
class _StubDPT:
    """Returns the (normalised) input mean + offset for every pixel."""

    def __init__(self, offset=0.0):
        import torch

        self._torch = torch
        self.offset = offset

    def __call__(self, x):
        torch = self._torch
        val = float(x.mean()) + self.offset
        out = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), val)
        return {"regression": out}


def test_predict_height_normalisation_blend_clamp_and_shape(monkeypatch):
    import torch

    from app.pipeline import rs_height

    device = torch.device("cpu")
    model = _StubDPT()
    # ImageNet stats on 0..255: mean of the three normalised channels at 128
    expected = ((128 - 123.675) / 58.395 + (128 - 116.28) / 57.12
                + (128 - 103.53) / 57.375) / 3.0

    rgb = np.full((30, 40, 3), 128, dtype="uint8")
    out = rs_height.predict_height(model, rgb, device=device, window=14, stride=7)
    assert out.shape == (30, 40) and out.dtype == np.float32
    assert abs(float(out.mean()) - expected) < 1e-3  # normalisation constants

    # input smaller than the window: reflection-padded, still fully covered
    small = rs_height.predict_height(model, rgb[:10, :12], device=device,
                                     window=14, stride=7)
    assert small.shape == (10, 12) and np.isfinite(small).all()

    clamped = rs_height.predict_height(_StubDPT(offset=-10.0), rgb, device=device,
                                       window=14, stride=7, clamp_min=0.0)
    assert clamped.min() == 0.0  # altimap's >= 0 m rule
    unclamped = rs_height.predict_height(_StubDPT(offset=-10.0), rgb, device=device,
                                         window=14, stride=7)
    assert unclamped.max() < 0.0  # RS3DAda path stays unclipped


def test_predict_height_resamples_1m_to_model_gsd_and_back(monkeypatch):
    import torch

    from app.pipeline import rs_height

    class ConstStub:
        def __init__(self, value=2.0):
            self._torch = torch
            self.value = value

        def __call__(self, x):
            return {"regression": self._torch.full(
                (x.shape[0], 1, x.shape[2], x.shape[3]), self.value)}

    # the window grid sizes expose the resolution the model actually ran at
    grids = []
    orig = rs_height._window_starts

    def recording(size, patch, stride):
        grids.append(size)
        return orig(size, patch, stride)

    monkeypatch.setattr(rs_height, "_window_starts", recording)

    device = torch.device("cpu")
    rgb = np.full((30, 40, 3), 90, dtype="uint8")
    # altimap target 0.33 m/px from a 1 m input: 30x40 -> 91x121 model grid
    out = rs_height.predict_height(ConstStub(2.0), rgb, device=device,
                                   window=14, stride=7, gsd_in=1.0,
                                   gsd_model=0.33)
    assert grids[0] == 91 and set(grids[1:]) == {121}   # height, then rows
    assert out.shape == (30, 40) and out.dtype == np.float32
    # output-scale convention: metres on BOTH grids, resampling never
    # rescales values (constant 2.0 m roundtrips unchanged)
    assert np.allclose(out, 2.0, atol=1e-4)


def test_predict_height_native_without_gsd_pair(monkeypatch):
    import torch

    from app.pipeline import rs_height

    class ConstStub:
        def __call__(self, x):
            return {"regression": torch.ones(
                (x.shape[0], 1, x.shape[2], x.shape[3]))}

    grids = []
    orig = rs_height._window_starts

    def recording(size, patch, stride):
        grids.append(size)
        return orig(size, patch, stride)

    monkeypatch.setattr(rs_height, "_window_starts", recording)

    device = torch.device("cpu")
    rgb = np.full((30, 40, 3), 90, dtype="uint8")
    for kwargs in ({}, {"gsd_in": 1.0}, {"gsd_model": 0.33}):
        grids.clear()
        out = rs_height.predict_height(ConstStub(), rgb, device=device,
                                       window=14, stride=7, **kwargs)
        assert grids[0] == 30 and set(grids[1:]) == {40}  # no resample (partial gsd)
        assert out.shape == (30, 40)


def test_infer_ndsm_metres_only_for_metric_backends(monkeypatch):
    class MetricStub:
        output_kind = "ndsm_metres"

        def infer_metres(self, rgb, gsd_in=None):
            return np.full(rgb.shape[:2], 3.5, dtype="float32")

    monkeypatch.setitem(depth._model_cache, "model", MetricStub())
    out = depth.infer_ndsm_metres(np.zeros((8, 9, 3), dtype="uint8"))
    assert out.shape == (8, 9) and np.allclose(out, 3.5)  # raw, no normalisation

    class RelativeStub:
        invert_depth = False

        def __call__(self, pil):
            return {"depth": np.zeros((4, 4), dtype="float32")}

    monkeypatch.setitem(depth._model_cache, "model", RelativeStub())
    with pytest.raises(depth.DepthModelError):
        depth.infer_ndsm_metres(np.zeros((4, 4, 3), dtype="uint8"))


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_rs_checkpoint_provenance_local_file(tmp_path):
    import hashlib

    w = tmp_path / "RS3DAda_vitl_DPT_height.pth"
    w.write_bytes(b"rs-weights")
    prov = depth.rs_checkpoint_provenance(
        str(w), "RS3DAda_vitl_DPT_height.pth", "weights MIT")
    assert prov["repo_id"] == str(w)
    assert prov["revision"] is None
    assert prov["weight_file"] == "RS3DAda_vitl_DPT_height.pth"
    assert prov["weight_sha256"] == hashlib.sha256(b"rs-weights").hexdigest()
    assert prov["license_note"] == "weights MIT"
    # explicit compute logging + pinned DINOv2 backbone code
    assert prov["compute"]["device"] in ("cpu", "cuda")
    assert prov["compute"]["source"] in ("auto", "env:DEPTHWIZARD_RS_DEVICE")
    assert isinstance(prov["compute"]["cuda_device_count"], int)
    assert set(prov["backbone_dinov2"]) >= {"zip_sha256", "revision", "cache_dir"}

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(depth.DepthWizardError):
        depth.rs_checkpoint_provenance(str(empty_dir), "best.pth")


# ---------------------------------------------------------------------------
# metric evaluation
# ---------------------------------------------------------------------------
def _ramp(shape=(64, 64), seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype("float32")
    return (y * 0.3 + x * 0.2 + rng.normal(0, 0.01, shape)).astype("float32")


def test_evaluate_metric_arrays_baselines_and_headlines():
    ref = _ramp()
    perfect = ref.copy()
    doubled = (2.0 * ref - 3.0).astype("float32")  # needs calibration to score
    record = evaluation.evaluate_metric_arrays(
        ref, {"metric": perfect, "relative": doubled},
        raw_headline={"metric"}, block=16, guard=4)

    _cal, eval_mask = evaluation.spatial_block_split(ref.shape, block=16, guard=4)
    eval_px = eval_mask & np.isfinite(ref)
    expected_zero = float(np.sqrt(np.mean(ref[eval_px].astype("float64") ** 2)))
    assert record["baselines"]["predict_zero"]["rmse"] == pytest.approx(
        expected_zero, rel=1e-6)

    m = record["backends"]["metric"]
    assert m["metric_source"] == "raw"
    assert m["metric_rmse"] < 1e-6                      # perfect arm
    assert m["delta_vs_predict_zero"] < 0.0
    assert m["improvement_vs_predict_zero"] == pytest.approx(1.0, rel=1e-6)
    assert m["scale"] == pytest.approx(1.0, rel=1e-3)   # drift diagnostic only

    r = record["backends"]["relative"]
    assert r["metric_source"] == "calibrated"
    assert r["scale"] == pytest.approx(0.5, rel=1e-3)
    assert r["offset"] == pytest.approx(1.5, abs=1e-3)
    assert r["metric_rmse"] is not None and r["metric_rmse"] < 0.05


def test_evaluate_metric_arrays_shared_mask_and_failure_counts():
    ref = _ramp(shape=(48, 48))
    holed = ref.copy()
    holed[0:4, :] = np.nan                                   # arm not scored there
    flat = np.full_like(ref, 5.0)                            # no local contrast
    record = evaluation.evaluate_metric_arrays(
        ref, {"holed": holed, "flat": flat}, raw_headline={"flat"},
        block=16, guard=4)
    assert record["n_valid"] < ref.size
    assert record["backends"]["holed"]["raw_rmse"] is not None  # scored around hole
    assert record["backends"]["holed"]["metric_source"] == "calibrated"

    flat_backend = record["backends"]["flat"]
    assert flat_backend["status"] in ("failed", "calibration_error")
    assert flat_backend["metric_rmse"] is not None           # raw headline stands
    assert flat_backend["metric_source"] == "raw"

    tile = {"status": "ok", "site_id": "s1", "backends": {"flat": flat_backend},
            "baselines": record["baselines"]}
    stats = evaluation._metric_failure_stats([tile], "flat")
    assert stats["valid"] == 1 and stats["failed"] == 0      # diagnostic-only fit


def test_aggregate_metric_records_skill_and_primary():
    def tile(site, rmse, zero, contaminated=()):
        return {"status": "ok", "site_id": site, "stratum": "urban",
                "contaminated_arms": list(contaminated),
                "baselines": {"predict_zero": {"status": "ok", "rmse": zero}},
                "backends": {"m": {"metric_rmse": rmse, "metric_source": "raw",
                                   "improvement_vs_predict_zero": 1 - rmse / zero,
                                   "raw_r": 0.5}}}
    records = [tile("a", 1.0, 4.0), tile("b", 2.0, 8.0)]
    agg = evaluation.aggregate_metric_records(records, bootstrap_iters=50)
    assert agg["primary_arm"] == "m"
    assert agg["arms"]["m"]["metric_rmse"] == pytest.approx(1.5)   # site-weighted
    assert agg["arms"]["m"]["skill_vs_predict_zero"] == pytest.approx(0.75)
    assert agg["per_stratum"]["urban"]["arms"]["m"]["metric_rmse"] == \
        pytest.approx(1.5)
    assert agg["bootstrap"]["m"]["metric_rmse"]["n_sites"] == 2

    records[0]["contaminated_arms"] = ["m"]
    agg2 = evaluation.aggregate_metric_records(records, bootstrap_iters=10)
    assert agg2["primary_arm"] is None           # excluded from the choice


def test_load_height_manifest_resolves_and_validates(tmp_path):
    (tmp_path / "ref.tif").write_bytes(b"x")
    mf = tmp_path / "height.csv"
    mf.write_text("tile_id,stratum,ndsm_path\n"
                  "t1,urban,ref.tif\n", encoding="utf-8")
    rows = evaluation.load_height_manifest(mf)
    assert rows[0]["ndsm_path"] == str((tmp_path / "ref.tif").resolve())

    bad = tmp_path / "bad.csv"
    bad.write_text("name,path\nt1,x\n", encoding="utf-8")
    with pytest.raises(evaluation.ManifestError):
        evaluation.load_height_manifest(bad)


def test_run_height_benchmark_ndsm_end_to_end(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    shape = (80, 80)
    ref = _ramp(shape=shape) + 10.0
    tif = tmp_path / "ndsm.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=shape[1], height=shape[0],
                       count=1, dtype="float32", crs="EPSG:32632",
                       transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(ref, 1)

    metric_dir = tmp_path / "run_metric" / "predictions"
    metric_dir.mkdir(parents=True)
    np.save(metric_dir / "t1_ndsm.npy", (ref + 0.1).astype("float32"))
    rel_dir = tmp_path / "run_rel" / "predictions"
    rel_dir.mkdir(parents=True)
    np.save(rel_dir / "t1_relative.npy", (0.5 * ref - 2.0).astype("float32"))

    mf = tmp_path / "height.csv"
    mf.write_text(f"tile_id,site_id,stratum,ndsm_path\nt1,s1,urban,{tif}\n",
                  encoding="utf-8")
    args = SimpleNamespace(
        command="benchmark", mode="ndsm", height_manifest=str(mf),
        predictions=[f"rarm={tmp_path / 'run_metric'}",
                     f"relarm={tmp_path / 'run_rel'}"],
        out=str(tmp_path / "out"), block=16, guard=4, calib_cap=100000,
        seed=42, bootstrap_iters=50)
    payload = evaluation.run_height_benchmark(args)

    agg = payload["aggregate"]
    assert "rarm" in agg["arms"] and "relarm" in agg["arms"]
    assert agg["arms"]["rarm"]["metric_source"] == "raw"
    assert agg["arms"]["relarm"]["metric_source"] == "calibrated"
    assert agg["arms"]["rarm"]["metric_rmse"] == pytest.approx(0.1, abs=0.02)
    assert "predict_zero" in agg["baselines"]
    assert (tmp_path / "out" / "height_benchmark.md").exists()
    saved = json.loads((tmp_path / "out" / "height_benchmark.json")
                       .read_text(encoding="utf-8"))
    assert saved["run"]["mode"] == "ndsm"
    assert saved["run"]["raw_headline_arms"] == ["rarm"]


def test_run_height_benchmark_dsm_composes_and_flags_licences(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    shape = (80, 80)
    prior = np.full(shape, 400.0, dtype="float32")
    ndsm = np.clip(_ramp(shape=shape, seed=11), 0, None)
    dsm = (prior + ndsm).astype("float32")

    def write(path, arr):
        with rasterio.open(path, "w", driver="GTiff", width=shape[1],
                           height=shape[0], count=1, dtype="float32",
                           crs="EPSG:32632",
                           transform=from_origin(0, 0, 1, 1)) as dst:
            dst.write(arr, 1)

    dsm_tif, prior_tif, ndsm_tif = (tmp_path / n for n in
                                    ("dsm.tif", "prior.tif", "ndsm.tif"))
    write(dsm_tif, dsm)
    write(prior_tif, prior)
    write(ndsm_tif, ndsm)

    metric_dir = tmp_path / "run_metric" / "predictions"
    metric_dir.mkdir(parents=True)
    np.save(metric_dir / "t1_ndsm.npy", (ndsm + 0.05).astype("float32"))
    rel_dir = tmp_path / "run_rel" / "predictions"
    rel_dir.mkdir(parents=True)
    np.save(rel_dir / "t1_relative.npy", (0.5 * ndsm - 2.0).astype("float32"))

    mf = tmp_path / "height.csv"
    mf.write_text("tile_id,site_id,stratum,ndsm_path,dsm_path,prior_path\n"
                  f"t1,s1,urban,{ndsm_tif},{dsm_tif},{prior_tif}\n",
                  encoding="utf-8")
    args = SimpleNamespace(
        command="benchmark", mode="dsm", height_manifest=str(mf),
        predictions=[f"altimap_v2={tmp_path / 'run_metric'}",
                     f"da2={tmp_path / 'run_rel'}"],
        out=str(tmp_path / "out"), block=16, guard=4, calib_cap=100000,
        seed=42, bootstrap_iters=50)
    payload = evaluation.run_height_benchmark(args)

    agg = payload["aggregate"]
    # arm = prior + predicted nDSM (rmse ~= the nDSM error, prior cancels out)
    assert agg["arms"]["altimap_v2"]["metric_rmse"] == pytest.approx(0.05, abs=0.02)
    assert "oracle" in agg["baselines"] and "prior_alone" in agg["baselines"]
    assert agg["baselines"]["prior_alone"]["rmse"] == pytest.approx(
        float(np.sqrt(np.mean(ndsm ** 2))), rel=0.05)
    # the non-commercial flag survives into the run payload
    note = payload["run"]["licenses"]["altimap_v2"]
    assert "NON-COMMERCIAL" in note


# ---------------------------------------------------------------------------
# tool registration
# ---------------------------------------------------------------------------
def test_force_backend_rs_clears_others(monkeypatch):
    _clear_all_backends(monkeypatch)
    monkeypatch.setattr(config, "DA3_SMALL_MODEL", "depth-anything/DA3-SMALL")
    monkeypatch.setattr(config, "TF_MODEL_PATH", "C:/tf/pix2pix")

    bt._force_backend("rs3dada", None)
    assert config.RS3DADA_MODEL == "JTRNEO/RS3DAda"
    assert config.DA3_SMALL_MODEL == "" and config.TF_MODEL_PATH == ""

    bt._force_backend("altimap_v2", "C:/snap/altimap-height")
    assert config.ALTIMAP_V2_MODEL == "C:/snap/altimap-height"
    assert config.RS3DADA_MODEL == ""


def test_model_info_rs_records_provenance_and_licence(monkeypatch):
    _clear_all_backends(monkeypatch)
    monkeypatch.setattr(config, "ALTIMAP_V2_MODEL", "Dilavesh/altimap-height")
    monkeypatch.setattr(depth, "_model_cache", {})
    calls = {}

    def fake_prov(model_id, filename, note=None):
        calls["args"] = (model_id, filename, note)
        return {"repo_id": model_id, "revision": "abc123", "snapshot": "/snap",
                "weight_file": filename, "weight_sha256": "f" * 64,
                "license_note": note,
                "compute": {"device": "cuda", "source": "auto",
                            "cuda_device_count": 1, "vram_cap": 0.6},
                "backbone_dinov2": {"zip_sha256": "a" * 64,
                                    "revision": "facebookresearch/dinov2@deadbeef",
                                    "cache_dir": "/hub/facebookresearch_dinov2_main"}}

    monkeypatch.setattr(depth, "rs_checkpoint_provenance", fake_prov)
    info = bt._model_info("altimap_v2")
    assert calls["args"][1] == "v2/best.pth"
    assert info["output_kind"] == "ndsm_metres"
    assert info["hf_revision"] == "abc123"
    assert info["file_sha256"] == {"v2/best.pth": "f" * 64}
    assert "NON-COMMERCIAL" in info["license_note"]
    assert info["compute"]["device"] == "cuda" and info["compute"]["vram_cap"] == 0.6
    assert info["backbone_dinov2"]["revision"].endswith("@deadbeef")


def test_prediction_suffix_metric_vs_relative():
    assert bt._prediction_suffix({"model": {"output_kind": "ndsm_metres"}}) == "_ndsm"
    assert bt._prediction_suffix({"model": {}}) == "_relative"
