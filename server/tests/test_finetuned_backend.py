"""Fourth depth backend (fine-tuned DA2-small + decoder head).

Covers DEPTHWIZARD_FINETUNED_MODEL: loads the checkpoint produced by
decoder_training/train_decoder.py, runs through the shared infer_relative_dsm
interface, and confirms correctly-shaped 0..1 output. Skipped until the
checkpoint exists (mirrors the IMELE skip-if-absent pattern).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from app import config
from app.pipeline import depth

_ENV_CKPT = os.environ.get("DEPTHWIZARD_FINETUNED_MODEL", "").strip()
_CKPT = Path(_ENV_CKPT) if _ENV_CKPT else (
    Path(__file__).resolve().parents[2] / "decoder_training"
    / "checkpoints" / "da2_decoder_finetuned.pt")


@pytest.mark.skipif(not _CKPT.exists(),
                    reason="da2_decoder_finetuned.pt not produced yet")
def test_finetuned_backend_loads_and_shaped_output(monkeypatch):
    monkeypatch.setattr(config, "FINETUNED_MODEL_PATH", str(_CKPT))
    model = depth._load_model()
    assert model.invert_depth is False
    rdsm = depth.infer_relative_dsm(
        np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
        ensure_real=True)
    assert rdsm.dtype == np.float32
    assert rdsm.shape == (64, 64)
    assert 0.0 <= float(rdsm.min()) <= float(rdsm.max()) <= 1.0