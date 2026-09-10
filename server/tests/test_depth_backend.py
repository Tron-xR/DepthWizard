"""Depth backends: env-configurable, fail cleanly when misconfigured.

Covers the three selectable backends:
- pix2pix GAN (TensorFlow SavedModel, default when models/pix2pix exists)
- IMELE (SENet154 + D2/MFF/R, PyTorch; opt-in via DEPTHWIZARD_IMELE_MODEL)
- Depth Anything V2 (transformers; TF_MODEL_PATH="" forces it)
"""
import os
from pathlib import Path

import pytest

from app import config
from app.pipeline import depth

_IMELE_CKPT = Path(__file__).resolve().parents[2] / "models" / "imele_model.tar"


def test_tf_backend_configured_but_broken_never_falls_back_to_hf(monkeypatch):
    monkeypatch.setattr(config, "TF_MODEL_PATH", "C:/nope/saved")
    with pytest.raises(Exception):
        depth._load_model()


def test_imele_backend_configured_but_broken_raises(monkeypatch):
    monkeypatch.setattr(config, "IMELE_MODEL_PATH", "C:/nope/imele.tar")
    with pytest.raises(Exception):
        depth._load_model()


@pytest.mark.skipif(not _IMELE_CKPT.exists(),
                    reason="IMELE checkpoint not downloaded (models/imele_model.tar)")
def test_imele_loads_and_produces_shaped_output():
    from app.pipeline import imele
    import torch

    model = imele.load_model(str(_IMELE_CKPT))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 32, 32)  # half-res output, as in upstream test.py