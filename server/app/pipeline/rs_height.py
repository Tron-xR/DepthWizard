"""Metric RS height backends (RS3DAda / altimap-height) - opt-in.

See docs/RS_HEIGHT_EVAL_DESIGN.md. These checkpoints have a raw linear
regression head that emits **nDSM in metres** (no activation, no 0..1
normalisation), so they stay off the relative-depth path: depth.py wraps this
module as an ``ndsm_metres`` backend and the benchmark scores the output
directly against a reference height raster.

Architecture code is vendored in rs_dpt.py (SynRS3D, MIT). Weights load
strictly: a partially-matched state dict would silently score random head
weights, so a mismatch is an error, not a warning.

Defaults (reported in every run): 518 px windows at half stride with a Hann
feather, bf16 autocast on CUDA. Upstream demos RS3DAda at 1022 px, but this
box has 4 GB VRAM; altimap's author runs the identical viT-L/DPT at 518 px on
an RTX 3050. TTA is off by default (documented deviation from both demos).

Compute device and VRAM cap are explicit (DEPTHWIZARD_RS_DEVICE,
DEPTHWIZARD_CUDA_MEM_FRACTION): never torch.cuda.is_available(), which can
report True with zero devices. GSD: when the caller knows the input GSD and
the checkpoint's target (altimap 0.33 m/px), predict_height resamples the RGB
to the target and the metre map back to the input grid.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import DepthWizardError

# Upstream export-path normalisation: RGB 0-255, ImageNet stats (no /255).
_MEAN = np.array([123.675, 116.28, 103.53], dtype="float32")[:, None, None]
_STD = np.array([58.395, 57.12, 57.375], dtype="float32")[:, None, None]
_PATCH = 518                    # multiple of 14; 1022 does not fit 4 GB
_STRIDE = _PATCH // 2
# Both published checkpoints are the same dual-head graph (height + 8-class
# OpenEarthMap segmentation trained jointly); only the regression head is used.
_HEADS = [{"name": "regression", "nclass": 1},
          {"name": "segmentation", "nclass": 8}]

_DEVICE_ENV = "DEPTHWIZARD_RS_DEVICE"
_VRAM_ENV = "DEPTHWIZARD_CUDA_MEM_FRACTION"
_LAST_COMPUTE: dict = {}


class RsHeightError(DepthWizardError):
    error_code = "rs_height_error"


def resolve_compute(explicit: Optional[str] = None) -> dict:
    """Decide the compute device and VRAM cap explicitly.

    No ``torch.cuda.is_available()`` anywhere: that call has been observed
    returning True with *zero* visible devices (empty CUDA_VISIBLE_DEVICES,
    torch 2.5.1/Windows), which then crashes deep inside autocast. Device
    presence is decided by ``torch.cuda.device_count()`` only.
    Precedence: ``explicit`` argument > DEPTHWIZARD_RS_DEVICE > auto
    (``cuda`` when device_count() > 0 else ``cpu``).
    """
    import torch

    env = os.environ.get(_DEVICE_ENV, "").strip()
    if explicit:
        device_str, source = explicit, "explicit"
    elif env:
        device_str, source = env, f"env:{_DEVICE_ENV}"
    else:
        device_str, source = None, "auto"

    count = torch.cuda.device_count()
    if device_str is None:
        device_str = "cuda" if count > 0 else "cpu"
    device = torch.device(device_str)
    if device.type == "cuda" and count == 0:
        raise RsHeightError(
            f"compute device {device_str!r} ({source}) requested but torch "
            f"sees 0 CUDA devices: set {_DEVICE_ENV}=cpu or unset it")

    raw = os.environ.get(_VRAM_ENV, "") or "0"
    try:
        fraction = float(raw)
    except ValueError:
        raise RsHeightError(f"{_VRAM_ENV} must be a number, got {raw!r}")
    cap = min(fraction, 1.0) if (fraction > 0 and device.type == "cuda") else None
    return {"device": device, "device_str": str(device), "source": source,
            "cuda_device_count": count, "vram_cap": cap}


def _apply_vram_cap(info: dict) -> None:
    """Set the per-process CUDA memory fraction (0/unset = no cap)."""
    if info["vram_cap"] is None:
        return
    import torch

    torch.cuda.set_per_process_memory_fraction(info["vram_cap"])


def compute_info() -> dict:
    """Serializable device/cap provenance: the settings the loaded model used,
    else what the next load would choose."""
    info = _LAST_COMPUTE or resolve_compute()
    return {"device": info["device_str"], "source": info["source"],
            "cuda_device_count": info["cuda_device_count"],
            "vram_cap": info["vram_cap"]}


def resolve_weight_path(model_id: str, filename: str) -> Path:
    """Local ``.pth`` path, ``dir/filename``, or an hf_hub_download.

    ``filename`` may contain a subdirectory (altimap v2 is ``v2/best.pth``:
    the repo has no revision tags, so the file path is the only version pin).
    """
    p = Path(model_id)
    if p.is_file():
        return p
    if p.is_dir():
        weight = p / filename
        if not weight.is_file():
            raise RsHeightError(f"no weight file '{filename}' under {model_id}")
        return weight
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id, filename))


def load_model(weight_path, *, device: Optional[str] = None):
    """Build the DPT graph, load the checkpoint strictly, return (model, device).

    ``device`` overrides DEPTHWIZARD_RS_DEVICE; both override auto-detection
    (device_count-based, see resolve_compute). The chosen settings are stashed
    for compute_info() so run provenance logs the device and VRAM cap used.
    """
    import torch

    info = resolve_compute(device)
    _LAST_COMPUTE.clear()
    _LAST_COMPUTE.update(info)
    _apply_vram_cap(info)
    torch_device = info["device"]

    from .rs_dpt import DPT_DINOv2

    model = DPT_DINOv2(encoder="vitl", head_configs=_HEADS, pretrained=False)
    model = model.to(torch_device).eval()

    state = torch.load(str(weight_path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RsHeightError(
            f"checkpoint does not match the known viT-L/DPT architecture "
            f"({weight_path}): {e}") from e
    return model, torch_device


def _window_starts(size: int, patch: int, stride: int) -> list:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch, stride))
    starts.append(size - patch)
    return starts


def _feather(patch: int, device, dtype):
    import torch

    ax = np.hanning(patch + 2)[1:-1].astype("float32") + 1e-3
    return torch.from_numpy(np.outer(ax, ax)).to(device=device, dtype=dtype)


def predict_height(model, rgb: np.ndarray, *, device, clamp_min: Optional[float] = None,
                   window: int = _PATCH, stride: int = _STRIDE,
                   gsd_in: Optional[float] = None,
                   gsd_model: Optional[float] = None) -> np.ndarray:
    """Windowed nDSM prediction in metres on the input pixel grid.

    Output is the raw regression head: no sign inversion, no normalisation.
    ``clamp_min`` applies a floor (altimap documents >= 0 m: height above
    ground is non-negative); RS3DAda's export path has no clip, so its caller
    passes None. Blanked edge tiles pass reflection padding so every pixel is
    covered by at least one full window.

    GSD handling (only when both values are given): the model is fed at
    ``gsd_model`` m/px - the RGB is bilinearly resampled by gsd_in/gsd_model
    before windowing (the documented 1 m Swiss tile -> 0.33 m altimap target
    is a 3.03x upsample), and the metre map is resampled back to the input
    grid (area-average when downscaling). Values stay metres on both grids;
    resampling never rescales them.
    """
    import torch

    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RsHeightError(f"expected (H,W,3) RGB, got shape {arr.shape}")
    h, w = arr.shape[:2]

    scale = 1.0
    if gsd_in and gsd_model and abs(gsd_in - gsd_model) > 0.02 * gsd_model:
        scale = gsd_in / gsd_model
    resized_h = max(1, round(h * scale)) if scale != 1.0 else h
    resized_w = max(1, round(w * scale)) if scale != 1.0 else w

    t = torch.from_numpy(
        arr.astype("float32").transpose(2, 0, 1)[None]).to(device)
    if scale != 1.0:
        t = torch.nn.functional.interpolate(t, size=(resized_h, resized_w),
                                            mode="bilinear", align_corners=False)
    t = (t - torch.as_tensor(_MEAN, device=device)) / torch.as_tensor(_STD, device=device)

    pad_h = max(0, window - resized_h)
    pad_w = max(0, window - resized_w)
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    big_h, big_w = t.shape[-2:]

    feather = _feather(window, device, t.dtype)
    acc = torch.zeros((big_h, big_w), dtype=torch.float32, device=device)
    wsum = torch.zeros_like(acc)

    use_amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_amp):
        for y in _window_starts(big_h, window, stride):
            for x in _window_starts(big_w, window, stride):
                out = model(t[:, :, y:y + window, x:x + window])["regression"]
                out = out[0, 0].float()
                acc[y:y + window, x:x + window] += out * feather
                wsum[y:y + window, x:x + window] += feather

    height = acc / wsum.clamp_min(1e-6)
    height = height[:resized_h, :resized_w]
    if scale != 1.0:
        height = torch.nn.functional.interpolate(
            height[None, None], size=(h, w),
            mode="area" if scale > 1 else "bilinear",
            align_corners=None if scale > 1 else False)[0, 0]
    height = height.cpu().numpy().astype("float32")
    if clamp_min is not None:
        np.maximum(height, clamp_min, out=height)
    return height
