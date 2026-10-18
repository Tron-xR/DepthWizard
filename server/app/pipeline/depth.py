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


# DA3 (Depth Anything 3) opt-in backends: (slug, config attribute, default HF
# repo id). Default is off (all config values empty); the deployment opts in by
# setting exactly one DEPTHWIZARD_DA3_*_MODEL env var to an HF repo id or a
# local snapshot directory. Licences: SMALL/BASE/MONO-LARGE Apache-2.0;
# LARGE/GIANT CC BY-NC 4.0 (non-commercial). The tool/eval CLI uses the
# defaults below when no --model-path override is given.
DA3_BACKENDS = (
    ("da3-small", "DA3_SMALL_MODEL", "depth-anything/DA3-SMALL"),
    ("da3-base", "DA3_BASE_MODEL", "depth-anything/DA3-BASE"),
    ("da3mono-large", "DA3MONO_LARGE_MODEL", "depth-anything/DA3MONO-LARGE"),
    ("da3-large", "DA3_LARGE_MODEL", "depth-anything/DA3-LARGE-1.1"),
    ("da3-giant", "DA3_GIANT_MODEL", "depth-anything/DA3-GIANT-1.1"),
)


def _selected_da3():
    """(slug, model_id) of the single configured DA3 backend, or None.

    More than one configured is an error: five silently-mixable candidates
    would make "which model actually ran" unanswerable from the artifacts.
    """
    selected = [(slug, getattr(config, attr))
                for slug, attr, _ in DA3_BACKENDS if getattr(config, attr)]
    if len(selected) > 1:
        raise DepthModelError(
            "multiple DA3 backends configured ("
            + ", ".join(slug for slug, _ in selected)
            + "): set exactly one DEPTHWIZARD_DA3_*_MODEL env var or none")
    return selected[0] if selected else None


# Remote-sensing height backends (opt-in, metric nDSM in metres; scored by the
# ndsm/dsm evaluation modes, see docs/RS_HEIGHT_EVAL_DESIGN.md). Same
# single-select contract as DA3. ``filename`` pins the weight file inside the
# repo (altimap has no revision tags - the path IS the version: root
# best.pth = v1, v2/best.pth = v2). Licences: weights MIT; altimap v2 trains
# in part on SynRS3D and its model card says non-commercial, so every output
# of that arm carries the note reported here.
RS_BACKENDS = (
    ("rs3dada", "RS3DADA_MODEL", "JTRNEO/RS3DAda",
     "RS3DAda_vitl_DPT_height.pth",
     "weights MIT; trained on SynRS3D (dataset licence disputed: HF card "
     "CC BY-NC 4.0 vs Zenodo CC BY 4.0) - see docs/RS_HEIGHT_EVAL_DESIGN.md"),
    ("altimap_v1", "ALTIMAP_V1_MODEL", "Dilavesh/altimap-height",
     "best.pth",
     "weights MIT; trained on GAMUS (CC BY 4.0)"),
    ("altimap_v2", "ALTIMAP_V2_MODEL", "Dilavesh/altimap-height",
     "v2/best.pth",
     "weights MIT but trained in part on SynRS3D; model card says "
     "NON-COMMERCIAL - keep this flag on every output of this arm"),
)


def _selected_rs():
    """(slug, model_id, filename, license_note) of the single configured RS
    backend, or None. More than one configured is an error (same reason as
    _selected_da3)."""
    selected = [(slug, getattr(config, attr), filename, note)
                for slug, attr, _repo, filename, note in RS_BACKENDS
                if getattr(config, attr)]
    if len(selected) > 1:
        raise DepthModelError(
            "multiple RS height backends configured ("
            + ", ".join(slug for slug, *_ in selected)
            + "): set exactly one DEPTHWIZARD_RS3DADA_MODEL / "
            "DEPTHWIZARD_ALTIMAP_V1_MODEL / DEPTHWIZARD_ALTIMAP_V2_MODEL "
            "env var or none")
    return selected[0] if selected else None


def get_depth_model() -> Optional[object]:
    """Return the cached model/pipeline or None if unavailable."""
    return _model_cache.get("model")


def backend_slug() -> str:
    """Short stable label for the ACTIVE backend, used to namespace artifact
    export directories (outputs/pix2pix, outputs/imele, ...). Derived from the
    loaded model's class so it can never drift from what actually ran; falls
    back to the config dispatch order used by _load_model when nothing is
    loaded yet.
    """
    model = _model_cache.get("model")
    if model is not None:
        name = type(model).__name__
        if name == "_TfSavedModel":
            return "pix2pix"
        if name == "_ImeleModelBackend":
            return "imele"
        if name == "_Da3ModelBackend":
            return model.slug
        if name == "_RsHeightBackend":
            return model.slug
        if name == "_FinetunedDepthBackend":
            return "finetuned"
        return "depth_anything"
    if config.FINETUNED_MODEL_PATH:
        return "finetuned"
    if config.IMELE_MODEL_PATH:
        return "imele"
    rs = _selected_rs()
    if rs is not None:
        return rs[0]
    da3 = _selected_da3()
    if da3 is not None:
        return da3[0]
    if config.TF_MODEL_PATH:
        return "pix2pix"
    return "depth_anything"


def model_identifier() -> str:
    """Human-readable identifier of whatever model is configured/loaded."""
    if config.FINETUNED_MODEL_PATH:
        return config.FINETUNED_MODEL_PATH
    if config.IMELE_MODEL_PATH:
        return config.IMELE_MODEL_PATH
    rs = _selected_rs()
    if rs is not None:
        return rs[1]
    da3 = _selected_da3()
    if da3 is not None:
        return da3[1]
    if config.TF_MODEL_PATH:
        return config.TF_MODEL_PATH
    return config.DEFAULT_DEPTH_MODEL


def active_output_kind() -> str:
    """'ndsm_metres' when the configured/loaded backend emits metric nDSM
    (RS height checkpoints), else 'relative' (every DA2/DA3/GAN/HF backend).

    Mirrors _load_model's dispatch order so callers (benchmark_tiles) know
    which inference entry point to use WITHOUT loading the model first.
    """
    model = _model_cache.get("model")
    if model is not None:
        return getattr(model, "output_kind", "relative")
    if not config.FINETUNED_MODEL_PATH and not config.IMELE_MODEL_PATH:
        if _selected_rs() is not None:
            return "ndsm_metres"
    return "relative"


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
        #
        # Point 8 (grid-to-grid resample): the 512² model grid -> native grid
        # (h,w) is a RESAMPLE of a continuous elevation field, not an alignment
        # problem - neither grid is georeferenced, the pair is pixel-matched
        # (model(h,w) == rDSM(h,w), same grid both sides). LANCZOS preserves
        # smoothness without worst-case mipmapping artifacts; if visible ringing
        # appears, Image.BILINEAR is a drop-in, but the GAN output is already
        # bounded so any artifact is far below the DEM's relief. Do not replace
        # with NEAREST (blocks on smooth elevation) or blindly upscale input RGB
        # to arbitrary sizes the U-Net cannot concatenate through.
        small = rgb
        if (h, w) != (512, 512):
            small = np.asarray(
                Image.fromarray(rgb.astype("uint8")).resize((512, 512), Image.LANCZOS),
                dtype="float32")
        batch = (small / 127.5 - 1.0)[None, ...]  # training norm: rgb/127.5 - 1
        out = self._infer(**{self._key: self._tf.constant(batch)})
        # RAW floating-point output: the ONLY transform below is squeeze().
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


class _Da3ModelBackend:
    """Depth Anything 3 (opt-in) adapter: DA3-SMALL/BASE/MONO-LARGE/LARGE/GIANT.

    Output convention: DA3's ``Prediction.depth`` is z-depth (larger = farther
    from the camera). Verified against the upstream code: the GLB exporter
    unprojects ``point = depth * ray(z=1)`` and the metric helper scales
    ``depth * focal / 300``. The harness convention, established by the DA2/HF
    adapter above (inverse depth -> inverted), is rDSM increasing with
    camera-to-ground distance, which a z-depth map already satisfies, so
    ``invert_depth = False`` for every DA3 size. That is the documented, fixed
    per-model rule decided from the model's definition - never chosen per tile
    or per result. The harness separately reports raw_r_opposite, so a wrong
    family-level call stays visible without silently switching signs.

    Depth comes back on DA3's processed grid (long side 504 by default), not
    the native tile grid; infer_relative_dsm resamples it (LANCZOS), exactly
    like the DA2/HF arm - process_res is deliberately left at the library
    default (no tuning on the test split). Imported lazily so servers without
    a configured DA3 backend never import torch or the DA3 package.
    """
    invert_depth = False

    def __init__(self, slug: str, model_id: str):
        import torch

        from depth_anything_3.api import DepthAnything3

        self.slug = slug
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self._model = DepthAnything3.from_pretrained(model_id).to(self.device)

    def __call__(self, pil):
        import torch

        rgb = np.asarray(pil.convert("RGB"), dtype="uint8")
        with torch.no_grad():
            prediction = self._model.inference([rgb])
        depth = np.asarray(prediction.depth, dtype="float32")
        if depth.ndim == 3:  # (N,H,W) -> (H,W)
            depth = depth[0]
        return {"depth": depth}


class _RsHeightBackend:
    """Metric RS height adapter (RS3DAda / altimap-height, opt-in).

    Unlike every other backend this one emits nDSM in METRES (raw linear
    regression head): ``output_kind = "ndsm_metres"`` marks it so the
    relative-depth inversion/normalization never touches it and
    ``infer_ndsm_metres`` is the only supported entry point. Native-grid
    windowed inference lives in rs_height.py (518 px windows / bf16 - the box
    has 4 GB VRAM; see docs/RS_HEIGHT_EVAL_DESIGN.md). ``license_note`` is
    carried on the instance so run artifacts can flag altimap v2's
    non-commercial training data. Imported lazily so servers without an RS
    backend never import torch.
    """
    output_kind = "ndsm_metres"
    invert_depth = False  # not a relative-depth model; stable contract for callers

    def __init__(self, slug: str, model_id: str, filename: str,
                 license_note: Optional[str] = None):
        from . import rs_height

        self.slug = slug
        self.model_id = model_id
        self.filename = filename
        self.license_note = license_note
        self.path = str(rs_height.resolve_weight_path(model_id, filename))
        self._model, self.device = rs_height.load_model(self.path)
        # altimap documents a 0.33 m/px target (GAMUS): callers pass gsd_in and
        # the RGB is resampled to this before inference. RS3DAda was trained on
        # 0.05-1.0 m bins, so a 1 m Swiss tile is native for it (no resample).
        self.model_gsd = 0.33 if slug.startswith("altimap") else None
        # altimap documents non-negative heights; RS3DAda's export path has no
        # clip, so it stays unclamped (per-checkpoint rule, never per-tile).
        self.clamp_min = 0.0 if slug.startswith("altimap") else None

    def infer_metres(self, rgb: np.ndarray,
                     gsd_in: Optional[float] = None) -> np.ndarray:
        from . import rs_height

        return rs_height.predict_height(self._model, rgb, device=self.device,
                                        clamp_min=self.clamp_min,
                                        gsd_in=gsd_in, gsd_model=self.model_gsd)

    def __call__(self, pil):
        return {"depth": self.infer_metres(np.asarray(pil.convert("RGB")))}


def rs_checkpoint_provenance(model_id: str, filename: str,
                             license_note: Optional[str] = None) -> dict:
    """Pinned weight provenance for one RS height checkpoint.

    Same contract as da3_checkpoint_provenance: ``repo_id``, ``revision``
    (hub commit sha the snapshot is pinned to, None for a local path/pickle),
    ``weight_file`` and ``weight_sha256``, plus the checkpoint's
    ``license_note`` (non-commercial flags must survive into run artifacts).
    Also records the compute settings the loaded model used (``compute``:
    device, device source, CUDA device count, VRAM cap fraction) and the
    pinned DINOv2 backbone code (``backbone_dinov2``: commit sha + zip
    sha256). Raises DepthModelError when the weight file is missing.
    """
    import hashlib
    from pathlib import Path

    from . import rs_dpt, rs_height

    weight = rs_height.resolve_weight_path(model_id, filename)
    if Path(model_id).exists():
        repo_id, revision = model_id, None
    else:
        repo_id = model_id
        revision = weight.resolve().parents[len(Path(filename).parts) - 1].name
    h = hashlib.sha256()
    with open(weight, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot": str(weight.parent),
        "weight_file": filename,
        "weight_sha256": h.hexdigest(),
        "license_note": license_note,
        "compute": rs_height.compute_info(),
        "backbone_dinov2": rs_dpt.dinov2_source_info(),
    }


def da3_checkpoint_provenance(model_id: str) -> dict:
    """Pinned weight provenance for one DA3 checkpoint.

    ``model_id`` is an HF repo id (resolved through the local HF cache via
    huggingface_hub, which downloads only missing files) or a local snapshot
    directory (the offline path). Returns ``repo_id``, ``revision`` (the hub
    commit sha the snapshot is pinned to - not a mutable branch name),
    ``weight_file`` and ``weight_sha256`` so a run names the exact bytes it
    scored. Raises DepthModelError when no weight file exists (an empty
    provenance must fail loudly, not degrade to "unknown").
    """
    import hashlib
    from pathlib import Path

    p = Path(model_id)
    if p.is_dir():
        snapshot, repo_id, revision = (p, model_id, None)
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(model_id))
        repo_id, revision = model_id, snapshot.name
    weights = sorted(snapshot.rglob("*.safetensors"))
    if not weights:
        raise DepthModelError(f"no .safetensors weight found under {snapshot}")
    weight = weights[0]
    h = hashlib.sha256()
    with open(weight, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot": str(snapshot),
        "weight_file": weight.name,
        "weight_sha256": h.hexdigest(),
    }


def _load_model():
    if config.FINETUNED_MODEL_PATH:
        return _FinetunedDepthBackend(config.FINETUNED_MODEL_PATH)

    if config.IMELE_MODEL_PATH:
        return _ImeleModelBackend(config.IMELE_MODEL_PATH)

    # Configured RS/DA3 outrank TF/HF: TF_MODEL_PATH is auto-discovered when
    # models/pix2pix exists, so an opt-in must be checked before it.
    rs = _selected_rs()
    da3 = _selected_da3()
    if rs is not None and da3 is not None:
        raise DepthModelError(
            f"both an RS height backend ({rs[0]}) and a DA3 backend ({da3[0]}) "
            "are configured: set exactly one DEPTHWIZARD_*_MODEL env var")
    if rs is not None:
        return _RsHeightBackend(rs[0], rs[1], rs[2], rs[3])
    if da3 is not None:
        return _Da3ModelBackend(*da3)

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


def infer_relative_dsm(rgb: np.ndarray, *, ensure_real: bool = True,
                       capture: Optional[dict] = None) -> np.ndarray:
    """Run depth inference on an RGB array (H,W,3 float or uint8) -> rDSM (H,W) float32.

    Returns a 0..1 normalized relative DSM (higher = brighter).

    capture:
        Optional dict; when given it is filled in-place with
        ``capture["raw"]``      = the model's raw floating-point output on the
                                  native input grid, BEFORE depth-sign
                                  inversion and BEFORE 0..1 normalization, and
        ``capture["relative"]`` = the same normalized rDSM returned here.
        Purely observational: no extra inference, no behavior change for
        callers that don't pass it. Used by the optional prediction-artifact
        export to expose intermediate stages without re-running the model.
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
    # Prefer the pipeline's NATIVE floating-point output when exposed. The HF
    # DepthEstimationPipeline's "depth" key is quantized to uint8 0..255 in its
    # postprocess - a real (if small) precision loss, measured on this pipeline:
    # 254 vs 255,111 unique rDSM levels, corr(float,uint8)=0.99996,
    # MAE=0.002, i.e. it does not explain weak calibration but is avoidable.
    # "predicted_depth" is the raw model tensor; the GAN/IMELE/finetuned
    # backends only ever return the float "depth" key, so they are unchanged.
    depth = out.get("predicted_depth")
    if depth is None:
        depth = out.get("depth")
    depth = np.asarray(depth, dtype="float32")
    # ensure 2D (H,W); squeeze any leading singleton band dims
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim > 2:
        depth = np.squeeze(depth)
    # Bring a model-grid tensor onto the input image grid (LANCZOS) so the
    # rest of the pipeline/calibration sees ONE pixel-aligned pair. Backends
    # that already produce native (h,w) arrays skip this resample.
    if depth.shape[:2] != (h, w):
        depth = np.asarray(Image.fromarray(depth).resize((w, h), Image.LANCZOS),
                           dtype="float32")
    # Stage A: raw model output on the native grid, still in the model's own
    # depth convention (not yet sign-flipped, not yet 0..1). Captured before
    # any transform so callers can inspect the true prediction.
    if capture is not None:
        capture["raw"] = depth.copy()
    # HF depth: predicted depth has nearer/bright as high; invert so brighter = higher.
    # Backends already yielding high=bright elev (e.g. the pix2pix GAN) skip this.
    if getattr(model, "invert_depth", True):
        depth = depth * -1.0
    rdsm = _normalize_01(depth)
    if capture is not None:
        capture["relative"] = rdsm.copy()
    return rdsm


def _normalize_01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo == 0:
        return np.zeros_like(x, dtype="float32")
    return ((x - lo) / (hi - lo)).astype("float32")


def infer_ndsm_metres(rgb: np.ndarray,
                      gsd_in: Optional[float] = None) -> np.ndarray:
    """Run a metric RS height backend -> nDSM in metres (H,W) float32.

    Strictly for ``ndsm_metres`` backends (RS3DAda / altimap-height): the raw
    regression head, NO depth-sign inversion and NO 0..1 normalization, on the
    input pixel grid. Raises for relative-depth backends - their output has no
    metric scale; use infer_relative_dsm for those.

    ``gsd_in`` = the input's GSD in m/px. When the checkpoint documents a
    target GSD (altimap: 0.33), the backend resamples the RGB to it and the
    metre map back to the input grid; RS3DAda (0.05-1.0 m training census)
    runs native regardless. Output values are metres on both grids.
    """
    from PIL import Image

    model = ensure_model()
    if getattr(model, "output_kind", None) != "ndsm_metres":
        raise DepthModelError(
            f"backend {backend_slug()!r} is relative-depth, not a metric nDSM "
            "model: infer_ndsm_metres needs one configured RS height backend "
            "(DEPTHWIZARD_RS3DADA_MODEL / DEPTHWIZARD_ALTIMAP_V1_MODEL / "
            "DEPTHWIZARD_ALTIMAP_V2_MODEL)")

    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    pil = Image.fromarray(arr.astype("uint8") if arr.max() <= 255 else arr)
    return model.infer_metres(np.asarray(pil), gsd_in=gsd_in)
