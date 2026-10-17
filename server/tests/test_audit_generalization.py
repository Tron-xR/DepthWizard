"""Focused tests for tools/audit_generalization.py pure helpers.

Covers the correlation/statistics helpers, the 31 px coarse/fine split, the
phase-randomization null field, and the FFT best-shift finder (synthetic rolls
that would catch sign, wrap, or boundary errors). No model, no real tiles.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = REPO_ROOT / "tools" / "audit_generalization.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("audit_generalization", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_generalization"] = mod
    spec.loader.exec_module(mod)
    return mod


ag = _load_tool()


def _smooth_field(seed: int = 0, size: int = 600) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.rand(size, size)
    for _ in range(20):
        base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0)
                + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5.0
    return 0.8 * base + 0.2 * rng.rand(size, size)


def test_pearson_bounds_and_constant():
    a = np.linspace(0, 1, 100)
    assert abs(ag._pearson(a, a) - 1.0) < 1e-12
    assert abs(ag._pearson(a, -a) + 1.0) < 1e-12
    assert np.isnan(ag._pearson(a, np.ones_like(a)))


def test_rdsm_stats_linspace():
    st = ag._rdsm_stats(np.linspace(0, 1, 101).reshape(101, 1))
    assert abs(st["std"] - 0.2915) < 0.01
    assert st["sat_frac"] == 2 / 101


def test_band_split_roundtrip():
    a = _smooth_field()
    assert np.allclose(ag._lowpass(a) + ag._highpass(a), a)


def test_phase_randomize_preserves_spectrum():
    a = _smooth_field(seed=2)
    n = ag._phase_randomize(a, np.random.RandomState(0))
    assert n.shape == a.shape
    f1 = np.abs(np.fft.rfft2(a))
    f2 = np.abs(np.fft.rfft2(n))
    assert np.allclose(f1, f2, rtol=1e-6, atol=1e-6)
    assert abs(ag._pearson(a, n)) < 0.3


def test_best_shift_finds_roll_and_beats_null():
    base = _smooth_field(seed=1)
    pred = np.roll(base, 2, axis=0)          # pred(y) == dem(y-2) -> dy=-2
    sh = ag.best_shift(pred, base)
    assert (sh["dx"], sh["dy"]) == (0, -2)
    assert sh["gain"] > 0.05
    assert sh["r_best"] > sh["r0"]
    assert abs(sh["r_transpose"]) < 0.2
    pred2 = np.roll(base, -3, axis=1)        # pred(x) == dem(x+3) -> dx=+3
    sh2 = ag.best_shift(pred2, base)
    assert (sh2["dx"], sh2["dy"]) == (3, 0)


def test_best_shift_null_gain_is_small():
    base = _smooth_field(seed=1)
    pred = np.roll(base, 2, axis=0)
    obs = ag.best_shift(pred, base)["gain"]
    null_g = ag.best_shift(
        pred, ag._phase_randomize(base, np.random.RandomState(1)))["gain"]
    assert null_g < obs
