"""Progressive Morphological Filter (Zhang et al., 2003) on a dense height raster.

Ground-filtering a DSM back toward bare earth. At each iteration a grayscale
morphological OPENING with an odd square structuring element suppresses sharp
upward features (buildings, trees) narrower than the current window; any pixel
that still rises more than a slope-and-window threshold above the opened
surface is declared non-ground and replaced with the opened value. Windows
double each iteration until they exceed a size relative to the raster extent.
Windows are sized in PIXELS from a real-world meter size using the raster's
own ground resolution, so the same knob works across input resolutions.

The final surface is the estimated bare-earth DEM: open-ground pixels are
untouched (opening of a smooth/linear surface equals the surface, so their
difference vs threshold is ~0), while object pixels are pulled down to the
local terrain envelope.

Operates on the float32 meter array the pipeline already produces (the DSM),
never on a point cloud: the project already has a full height grid per pixel.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage


def _n_pixels(window_m: float, cell_size_m: float) -> int:
    """Smallest odd window (>=3 px) covering at least window_m on the ground."""
    px = max(3, int(round(window_m / cell_size_m)))
    if px % 2 == 0:
        px += 1
    return px


def progressive_morphological_filter(
    height: np.ndarray,
    cell_size_m: float,
    *,
    slope_threshold: float = 0.3,
    dh0: float = 0.3,
    start_window_m: float = 3.0,
    max_window_m: Optional[float] = None,
    max_window_ratio: float = 0.25,
) -> np.ndarray:
    """Bare-earth DEM estimate from a DSM height raster (Zhang et al. 2003).

    Parameters (tuned against synthetic + prototype testing, see
    tests/test_dem_filter.py for the concrete before/after values):
      slope_threshold: per-meter vertical tolerance for how much a feature may
        out-climb the opened surface as the window grows (0.3 means 0.3 m of
        allowed rise per 1 m of window ground-spacing).
      dh0:          0.3 m base tolerance applied even at the first (smallest)
        window, so 1-2 cm terrain texture is never flagged.
      start_window_m / doubling: 3 m -> 6 -> 12 -> ... so features up to a few
        windows wide are removed while broad terrain (mountains/valleys are
        far larger than any window here) survives opening unchanged.
      max_window_ratio: stop once the window radius exceeds this FRACTION of
        the raster's longest side (25% by default); a building larger than
        ~1/4 of the tile is really terrain.

    Returns a float32 array of the same shape as height.
    """
    h, w = height.shape
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")
    if max_window_m is None:
        max_window_m = max(h, w) * cell_size_m * max_window_ratio
    max_window_pix = _n_pixels(max_window_m, cell_size_m)

    work = height.astype("float64").copy()
    window_m = start_window_m
    while True:
        w_pix = _n_pixels(window_m, cell_size_m)
        if w_pix > max_window_pix:
            break
        opened = ndimage.grey_opening(work, size=(w_pix, w_pix))
        # Zhang's slope threshold: dh_max grows linearly with the window's
        # ground size (w_pix * cell_size_m), plus the dh0 base tolerance.
        dh_max = slope_threshold * (w_pix * cell_size_m) + dh0
        mask = (work - opened) > dh_max
        if mask.any():
            work[mask] = opened[mask]
        window_m *= 2.0
    return work.astype("float32")
