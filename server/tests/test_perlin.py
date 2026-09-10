"""Deterministic synthetic-terrain generator is stable and ridge-producing."""
import numpy as np

from app.pipeline import perlin


def test_generate_dem_deterministic_full_range():
    a = perlin.generate_dem(256, seed=3)
    b = perlin.generate_dem(256, seed=3)
    assert np.array_equal(a, b)
    assert a.min() < 0.05 and a.max() > 0.95


def test_generate_dem_not_a_ramp():
    dem = perlin.generate_dem(256, seed=3)
    blur = np.ones((17, 17)) / 289.0
    pad = np.pad(dem, 8, mode="edge")
    smooth = sum(blur[i, j] * pad[i:i + 256, j:j + 256]
                 for i in range(17) for j in range(17))
    assert float((dem - smooth).std()) > 0.01


def test_render_rgb_shape():
    assert perlin.render_rgb(perlin.generate_dem(64)).shape == (64, 64, 3)