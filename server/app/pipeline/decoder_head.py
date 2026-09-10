"""Small trainable decoder head for the fine-tuned Depth Anything V2 backend.

Additive-only module: shared by the OFFLINE training pipeline
(decoder_training/train_decoder.py) and the production fourth backend
(_FinetunedDepthBackend in app/pipeline/depth.py) so checkpoint state_dict keys
always match. Does not touch any existing backend behavior.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MODEL_ID = "LiheYoung/depth-anything-small-hf"

_FALLBACK_MEAN = (0.485, 0.456, 0.406)
_FALLBACK_STD = (0.229, 0.224, 0.225)


class DecoderHead(nn.Module):
    """4-layer conv stack refining the backbone's raw depth map -> elevation.

    Output is a monotone prediction at the same (1,1,H,W) resolution as the
    input depth; downstream _normalize_01 in depth.py scales it into 0..1
    (brighter = higher), matching the project's rDSM convention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        return self.net(depth)


def init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def to_pixel_values(rgb: np.ndarray, proc, size: int | None = None) -> torch.Tensor:
    """Encode a uint8 (H,W,3) RGB array into model pixel_values (3,H,W).

    Mirrors AutoImageProcessor semantics for this model: ImageNet mean/std on a
    /255 rescaled input. `size` optionally resizes first (LANCZOS). The batch
    dim is added by the caller (DataLoader collation stacks (3,H,W) samples;
    the backend unsqueezes for a single-image call). Fully-conv backbone
    accepts any resolution, so size may stay None when the input is already
    the training resolution.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("to_pixel_values expects uint8 RGB")
    if size is not None and (rgb.shape[0] != size or rgb.shape[1] != size):
        from PIL import Image

        rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.LANCZOS))

    mean = np.asarray(getattr(proc, "image_mean", None) or _FALLBACK_MEAN,
                      dtype="float32")
    std = np.asarray(getattr(proc, "image_std", None) or _FALLBACK_STD,
                     dtype="float32")
    x = rgb.astype("float32") / 255.0
    x = (x - mean[None, None, :]) / std[None, None, :]
    return torch.from_numpy(x.transpose(2, 0, 1)).float()