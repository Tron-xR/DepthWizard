"""Synthetic terrain for training/visualization data.

Borrows the RandomDEM idea from Panagiotou et al.'s ImageToDEM repo: Perlin-style
fractional Brownian motion DEMs, rendered into hillshaded RGB images that can be
paired with ground-truth elevation for training/eval data. Numpy + PIL only,
fully seeded and deterministic (unlike the original's unseeded loop).
"""
from __future__ import annotations

import argparse

import numpy as np


def _upsample(grid: np.ndarray, size: int) -> np.ndarray:
    cols = np.linspace(0, grid.shape[1] - 1, size)
    rows = np.linspace(0, grid.shape[0] - 1, size)
    widen = np.apply_along_axis(
        lambda col: np.interp(cols, np.arange(len(col)), col), 0, grid)
    return np.apply_along_axis(
        lambda row: np.interp(rows, np.arange(len(row)), row), 1, widen)


def _fbm(size: int, octaves: int, rng: np.random.Generator) -> np.ndarray:
    z = np.zeros((size, size))
    amp = 1.0
    norm = 0.0
    for o in range(octaves):
        gres = min(2 ** (o + 2), size)
        z += amp * _upsample(rng.random((gres, gres)), size)
        norm += amp
        amp *= 0.5
    return z / norm


def generate_dem(size: int = 512, *, seed: int = 7, octaves: int = 5,
                 ridge_weight: float = 0.45) -> np.ndarray:
    """Deterministic fractional-Brownian DEM in float 0..1 (higher = brighter)."""
    base = _fbm(size, octaves, np.random.default_rng(seed))
    ridge = 1.0 - np.abs(2.0 * _fbm(size, octaves, np.random.default_rng(seed + 1)) - 1.0)
    dem = (1 - ridge_weight) * base + ridge_weight * ridge
    lo, hi = float(dem.min()), float(dem.max())
    return (dem - lo) / (hi - lo)


def hillshade(dem: np.ndarray, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    """Standard hillshade (ESRI-ish): light from az/alt degrees, output 0..1."""
    az, alt = np.radians(az), np.radians(alt)
    gy, gx = np.gradient(dem)
    df = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(
        (np.cos(az) * gx + np.sin(az) * gy) * -np.sin(alt) + np.cos(alt),
        0, 1) / np.sqrt(1.0 + df ** 2)


def render_rgb(dem: np.ndarray, *, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    """Banded, tinted, hillshaded RGB render of a DEM -> H,W,3 uint8."""
    h, w = dem.shape
    rgb = np.empty((h, w, 3))
    for c in range(3):
        rgb[:, :, c] = np.where(dem < 0.30, 0.05 + 0.30 * dem,
                        np.where(dem < 0.55, 0.12 + 0.36 * dem,
                        np.where(dem < 0.80, 0.20 + 0.26 * dem,
                                         0.26 + 0.16 * dem)))
    tint = np.where(dem < 0.30, 0.45,
           np.where(dem < 0.55, 0.80,
           np.where(dem < 0.80, 1.10, 0.90)))
    rgb[..., 0] *= tint
    rgb[..., 1] *= tint * 1.12
    rgb[..., 2] *= tint * 0.80
    rgb *= hillshade(dem, az, alt)[..., None]
    return np.clip(rgb * 255, 0, 255).astype("uint8")


def _self_check() -> None:
    a = generate_dem(256, seed=3)
    b = generate_dem(256, seed=3)
    assert np.array_equal(a, b), "not deterministic"
    assert a.min() < 0.05 and a.max() > 0.95, "range not full"
    blur = np.ones((17, 17)) / 289.0
    pad = np.pad(a, 8, mode="edge")
    smooth = np.zeros_like(a)
    for i in range(17):
        for j in range(17):
            smooth += blur[i, j] * pad[i:i + a.shape[0], j:j + a.shape[1]]
    assert float((a - smooth).std()) > 0.01, "DEM degenerated into a ramp/slab"
    assert render_rgb(a).shape == (256, 256, 3)


if __name__ == "__main__":
    _self_check()
    ap = argparse.ArgumentParser(description="Deterministic synthetic terrain")
    ap.add_argument("--out", help="write hillshaded RGB PNG here")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--npz", help="also save a (1,size,size) DEM npz for training")
    args = ap.parse_args()
    p = argparse.Namespace(**vars(args)) if False else args
    assert p.size in (256, 512, 1024), "size must be a power of two"
    dem = generate_dem(p.size, seed=p.seed)
    print("dem stats: min %.3f mean %.3f max %.3f" %
          (dem.min(), dem.mean(), dem.max()))
    if p.npz:
        np.savez_compressed(p.npz, dem[None, ...].astype("float32"))
        print("wrote", p.npz)
    if p.out:
        from PIL import Image
        Image.fromarray(render_rgb(dem), "RGB").save(p.out)
        print("wrote", p.out)