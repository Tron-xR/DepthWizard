"""Backbone evaluation: DA3 registration/sign handling + control arms.

Covers the Phase-2 backbone-eval additions:
- DA3 backends are strictly opt-in and single-select (depth._selected_da3)
- the DA3 sign rule (invert_depth = False: z-depth is already far-bright) is
  honored end-to-end by infer_relative_dsm, and the flip still works for
  near-bright backends
- da3_checkpoint_provenance hashes local snapshots, never degrades silently
- the trivial control arms (brightness/sobel) are normalized and sensitive
- tools/benchmark_tiles registers the new backends (force/model_info)
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from app import config
from app.pipeline import depth

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = REPO_ROOT / "tools" / "benchmark_tiles.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("benchmark_tiles", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_tiles"] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load_tool()


def _clear_da3(monkeypatch):
    for _, attr, _default in depth.DA3_BACKENDS:
        monkeypatch.setattr(config, attr, "")


def _clear_other_backends(monkeypatch):
    monkeypatch.setattr(config, "FINETUNED_MODEL_PATH", "")
    monkeypatch.setattr(config, "IMELE_MODEL_PATH", "")
    monkeypatch.setattr(config, "TF_MODEL_PATH", "")


# ---------------------------------------------------------------------------
# DA3 opt-in selection
# ---------------------------------------------------------------------------
def test_selected_da3_is_off_by_default_and_single_select(monkeypatch):
    _clear_da3(monkeypatch)
    assert depth._selected_da3() is None

    monkeypatch.setattr(config, "DA3_SMALL_MODEL", "depth-anything/DA3-SMALL")
    assert depth._selected_da3() == ("da3-small", "depth-anything/DA3-SMALL")

    monkeypatch.setattr(config, "DA3_BASE_MODEL", "depth-anything/DA3-BASE")
    with pytest.raises(depth.DepthModelError):
        depth._selected_da3()


def test_model_identifier_reports_configured_da3(monkeypatch):
    _clear_other_backends(monkeypatch)
    _clear_da3(monkeypatch)
    monkeypatch.setattr(config, "DA3MONO_LARGE_MODEL", "C:/snap/DA3MONO-LARGE")
    assert depth.model_identifier() == "C:/snap/DA3MONO-LARGE"
    assert depth.backend_slug() == "da3mono-large"


# ---------------------------------------------------------------------------
# sign convention
# ---------------------------------------------------------------------------
def test_da3_backend_declares_z_depth_convention():
    # z-depth (larger = farther) already satisfies the harness convention
    # (rDSM increases with camera-to-ground distance), so no inversion.
    assert depth._Da3ModelBackend.invert_depth is False


def test_infer_relative_dsm_sign_follows_invert_depth(monkeypatch):
    class FarBrightStub:
        invert_depth = False

        def __call__(self, pil):
            ramp = np.tile(np.linspace(1.0, 10.0, 20, dtype="float32"), (20, 1))
            return {"depth": ramp}

    monkeypatch.setitem(depth._model_cache, "model", FarBrightStub())
    rdsm = depth.infer_relative_dsm(np.zeros((40, 40, 3), dtype="uint8"))
    assert rdsm.shape == (40, 40)  # DA3 grid is resampled back onto the input
    assert rdsm[0, -1] > rdsm[0, 0]  # far (right) stays bright: no flip

    class NearBrightStub(FarBrightStub):
        invert_depth = True

    monkeypatch.setitem(depth._model_cache, "model", NearBrightStub())
    flipped = depth.infer_relative_dsm(np.zeros((40, 40, 3), dtype="uint8"))
    assert flipped[0, -1] < flipped[0, 0]  # near-bright backends are inverted


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_da3_checkpoint_provenance_hashes_local_snapshot(tmp_path):
    snap = tmp_path / "DA3-SMALL"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"fake-weights")
    prov = depth.da3_checkpoint_provenance(str(snap))
    assert prov["repo_id"] == str(snap)
    assert prov["revision"] is None
    assert prov["weight_file"] == "model.safetensors"
    assert prov["weight_sha256"] == hashlib.sha256(b"fake-weights").hexdigest()

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(depth.DepthModelError):
        depth.da3_checkpoint_provenance(str(empty))


# ---------------------------------------------------------------------------
# control arms
# ---------------------------------------------------------------------------
def test_control_arms_normalized_and_sensitive():
    grad = np.tile(np.arange(32, dtype="uint8")[None, :], (16, 1))
    rgb = np.stack([grad, grad, grad], axis=-1)
    bright = bt._control_infer("brightness")(rgb)
    assert bright.min() == 0.0 and bright.max() == 1.0
    assert bright[0, 0] < bright[0, -1]

    edge = np.zeros((16, 32), dtype="uint8")
    edge[:, 16:] = 255
    rgb_edge = np.stack([edge, edge, edge], axis=-1)
    sob = bt._control_infer("sobel")(rgb_edge)
    assert sob.min() == 0.0 and sob.max() == 1.0
    assert sob[8, 15] == sob.max() and sob[8, 16] == sob.max()  # step edge
    assert sob[8, 0] == sob.min()

    flat = bt._control_infer("brightness")(np.full((8, 8, 3), 7, dtype="uint8"))
    assert flat.min() == flat.max() == 0.0  # constant input: zeros, no NaN


# ---------------------------------------------------------------------------
# tool registration
# ---------------------------------------------------------------------------
def test_model_info_control_has_no_checkpoint():
    info = bt._model_info("brightness")
    assert info == {"slug": "brightness", "identifier": "control:brightness",
                    "file_sha256": {}}


def test_force_backend_da3_selects_exactly_one(monkeypatch):
    _clear_da3(monkeypatch)
    _clear_other_backends(monkeypatch)
    monkeypatch.setattr(config, "TF_MODEL_PATH", "C:/tf/pix2pix")
    monkeypatch.setattr(config, "DA3_BASE_MODEL", "depth-anything/DA3-BASE")

    bt._force_backend("da3-small", None)
    assert config.DA3_SMALL_MODEL == "depth-anything/DA3-SMALL"
    assert config.DA3_BASE_MODEL == ""  # forcing clears every other selection
    assert config.TF_MODEL_PATH == ""

    bt._force_backend("da3-large", "C:/snap/DA3-LARGE-1.1")
    assert config.DA3_LARGE_MODEL == "C:/snap/DA3-LARGE-1.1"
    assert config.DA3_SMALL_MODEL == ""


def test_model_info_da3_records_hub_provenance(monkeypatch):
    _clear_da3(monkeypatch)
    _clear_other_backends(monkeypatch)
    monkeypatch.setattr(config, "DA3_SMALL_MODEL", "depth-anything/DA3-SMALL")
    monkeypatch.setattr(depth, "_model_cache", {})
    monkeypatch.setattr(depth, "da3_checkpoint_provenance",
                        lambda model_id: {"repo_id": model_id, "revision": "abc123",
                                          "snapshot": "/snap", "weight_file": "model.safetensors",
                                          "weight_sha256": "f" * 64})
    info = bt._model_info("da3-small")
    assert info["slug"] == "da3-small"
    assert info["identifier"] == "depth-anything/DA3-SMALL"
    assert info["hf_repo_id"] == "depth-anything/DA3-SMALL"
    assert info["hf_revision"] == "abc123"
    assert info["file_sha256"] == {"model.safetensors": "f" * 64}
