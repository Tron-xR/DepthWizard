"""Monocular depth backbone + rDSM (relative elevation) extraction.

Uses a pretrained Depth Anything V2 model served through HuggingFace
transformers. The model predicts *relative depth* (inverted: nearby is bright);
we invert and normalize into a 0..1 relative DSM where brighter = higher,
matching what the Unity mesh generator expects.

Model loading is lazy and cached globally so the server starts fast. If the
model cannot be downloaded (e.g. no network on first run), a configurable
fallback produces a placeholder surface so the rest of the pipeline/testable
logic remains usable.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from .. import config
from ..errors import DepthWizardError

_model_lock = threading.Lock()
_model_cache: dict = {}


class DepthModelError(DepthWizardError):
    error_code = "depth_model_error"


def get_depth_model() -> Optional[object]:
    """Return the cached model/pipeline or None if unavailable."""
    return _model_cache.get("model")


class _TfSavedModel:
    """Adapter so a pix2pix SavedModel satisfies the depth-backbone contract.

    Output is the generator's tanh in [-1,1] with brighter = higher elevation
    (it was trained to mimic the normalized DEM), so no sign inversion is
    applied downstream (invert_depth=False). Accepts variable-size input, so
    inference runs at native resolution with no 256px downscale.
    """
    invert_depth = False

    def __init__(self, path: str):
        import tensorflow as tf  # lazy: unconfigured servers never import TF

        self._tf = tf
        signatures = tf.saved_model.load(path).signatures
        if "serving_default" not in signatures:
            raise DepthModelError(f"No serving_default signature in SavedModel: {path}")
        self._infer = signatures["serving_default"]
        try:
            self._key = next(iter(self._infer.structured_input_signature[1]))
        except Exception as e:
            raise DepthModelError(f"Cannot inspect SavedModel input signature: {e}") from e

    def __call__(self, pil):
        from PIL import Image

        rgb = np.asarray(pil.convert("RGB"), dtype="float32")
        h, w = rgb.shape[:2]
        # The U-Net's skip concats only work on sizes that divide cleanly; run
        # the proven-safe fixed grid and resample the prediction back.
        # ponytail: fixed 512 grid; upgrade path = pad to a multiple or retrain
        # the model with variable-size support.
        small = rgb
        if (h, w) != (512, 512):
            small = np.asarray(
                Image.fromarray(rgb.astype("uint8")).resize((512, 512), Image.LANCZOS),
                dtype="float32")
        batch = (small / 127.5 - 1.0)[None, ...]  # training norm: rgb/127.5 - 1
        out = self._infer(**{self._key: self._tf.constant(batch)})
        dem = np.squeeze(list(out.values())[0].numpy()).astype("float32")
        if dem.shape != (h, w):
            dem = np.asarray(Image.fromarray(dem).resize((w, h), Image.LANCZOS),
                             dtype="float32")
        return {"depth": dem}


class _ImeleModelBackend:
    """Adapter so the IMELE PyTorch backbone satisfies the same contract.

    The IMELE checkpoint (SENet154 + D2/MFF/R) is a fixed 440x440 model trained
    to regress building height (brighter = higher), so invert_depth=False and
    the same fixed-grid/resample pattern as the pix2pix adapter is used. The
    checkpoint's legacy E.Harm.* keys are dropped on load (see app/pipeline/
    imele.py). Imported lazily so servers never configured for IMELE don't pay
    the torch import cost at startup.
    """
    invert_depth = False

    def __init__(self, path: str):
        from .imele import load_model

        self._model = load_model(path)

    def __call__(self, pil):
        import torch
        from PIL import Image

        from .imele import CROP, _IMAGENET_MEAN, _IMAGENET_STD

        rgb = np.asarray(pil.convert("RGB"), dtype="float32")
        h, w = rgb.shape[:2]
        small = rgb
        if (h, w) != (CROP, CROP):
            small = np.asarray(
                Image.fromarray(rgb.astype("uint8")).resize((CROP, CROP), Image.LANCZOS),
                dtype="float32")
        t = torch.from_numpy(small.transpose(2, 0, 1)[None, ...]).float() / 255.0
        mean = torch.tensor(_IMAGENET_MEAN)[:, None, None]
        std = torch.tensor(_IMAGENET_STD)[:, None, None]
        t = (t - mean) / std
        with torch.no_grad():
            out = self._model(t)  # (1,1,H/2,W/2) at the model's working res
            out = torch.nn.functional.interpolate(out, size=(CROP, CROP),
                                                  mode="bilinear", align_corners=False)
        dem = out[0, 0].cpu().numpy().astype("float32")
        if dem.shape != (h, w):
            dem = np.asarray(Image.fromarray(dem).resize((w, h), Image.LANCZOS),
                             dtype="float32")
        return {"depth": dem}


class _FinetunedDepthBackend:
    """Fourth backend: frozen DA2-small backbone + fine-tuned decoder head.

    Trained (decoder_training/train_decoder.py) to produce brighter=higher
    elevation against per-tile normalized DEMs, so invert_depth=False and no
    sign inversion is applied. Fully-conv, so native-resolution inference with
    no fixed-grid downscale. Imported lazily so servers never configured for it
    don't pay the torch/transformers import cost at startup.
    """
    invert_depth = False

    def __init__(self, ckpt: str):
        import torch
        from transformers import (AutoImageProcessor,
                                  AutoModelForDepthEstimation)

        from . import decoder_head

        self.proc = AutoImageProcessor.from_pretrained(decoder_head.MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            decoder_head.MODEL_ID)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.model.to(self.device)
        self.head = decoder_head.DecoderHead().to(self.device)
        state = torch.load(ckpt, map_location=self.device)
        self.head.load_state_dict(state["state_dict"])
        self.head.eval()

    def __call__(self, pil):
        import numpy as np
        import torch
        from PIL import Image

        from . import decoder_head

        rgb = np.asarray(pil.convert("RGB"), dtype="uint8")
        h, w = rgb.shape[:2]
        with torch.no_grad():
            pv = decoder_head.to_pixel_values(rgb, self.proc).to(self.device)
            depth = self.model(pixel_values=pv.unsqueeze(0)).predicted_depth
            dem = self.head(depth)[0, 0].cpu().numpy().astype("float32")
        if dem.shape != (h, w):
            dem = np.asarray(Image.fromarray(dem).resize((w, h), Image.LANCZOS),
                             dtype="float32")
        return {"depth": dem}


def _load_model():
    if config.FINETUNED_MODEL_PATH:
        return _FinetunedDepthBackend(config.FINETUNED_MODEL_PATH)

    if config.IMELE_MODEL_PATH:
        return _ImeleModelBackend(config.IMELE_MODEL_PATH)

    if config.TF_MODEL_PATH:
        return _TfSavedModel(config.TF_MODEL_PATH)

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    if config.DEFAULT_DEPTH_MODEL.startswith("hf://") or "/" in config.DEFAULT_DEPTH_MODEL:
        pipe = pipeline(
            "depth-estimation",
            model=config.DEFAULT_DEPTH_MODEL,
            device=device,
        )
    else:  # allow raw torch hub names
        pipe = pipeline("depth-estimation", model=config.DEFAULT_DEPTH_MODEL)
    pipe.invert_depth = True
    return pipe


def ensure_model() -> object:
    """Load (once) and return the depth model. Raises on failure."""
    with _model_lock:
        if "model" in _model_cache:
            return _model_cache["model"]
        try:
            _model_cache["model"] = _load_model()
        except Exception as e:
            raise DepthModelError(f"Failed to load depth model: {e}") from e
        return _model_cache["model"]


def _fallback_rdsm(height: int, width: int) -> np.ndarray:
    """A deterministic placeholder surface used when the model can't load.

    Produces a southwest-low / northeast-high gradient so the 3D view is still
    navigable. Clearly not a real DSM; the UI should flag this.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype("float32")
    return (xx / max(width - 1, 1) * 0.5 + yy / max(height - 1, 1) * 0.5)


def infer_relative_dsm(rgb: np.ndarray, *, ensure_real: bool = True) -> np.ndarray:
    """Run depth inference on an RGB array (H,W,3 float or uint8) -> rDSM (H,W) float32.

    Returns a 0..1 normalized relative DSM (higher = brighter).
    """
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[2] == 3:
        # assume uint8 0..255
        arr = arr.copy()
    elif arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w = arr.shape[:2]

    try:
        model = ensure_model()
    except DepthModelError:
        if ensure_real:
            raise
        return _fallback_rdsm(h, w)

    pil = Image.fromarray(arr.astype("uint8") if arr.max() <= 255 else arr)
    # Model outputs inverse depth; invert so higher = higher elevation.
    out = model(pil)
    depth = out.get("depth")
    if depth is None and "predicted_depth" in out:
        depth = out["predicted_depth"]
    depth = np.asarray(depth, dtype="float32")
    # HF depth: predicted depth has nearer/bright as high; invert so brighter = higher.
    # Backends already yielding high=bright elev (e.g. the pix2pix GAN) skip this.
    if getattr(model, "invert_depth", True):
        depth = depth * -1.0
    # ensure 2D (H,W); squeeze any leading singleton band dims
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    return _normalize_01(depth)


def _normalize_01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo == 0:
        return np.zeros_like(x, dtype="float32")
    return ((x - lo) / (hi - lo)).astype("float32")
