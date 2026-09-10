"""Unit tests for the accuracy-evaluation module."""
import numpy as np

from app.pipeline import validate as vmod


def test_compute_metrics():
    pred = np.array([1, 2, 3, 4], dtype="float32")
    ref = np.array([1, 3, 3, 5], dtype="float32")
    m = vmod.compute_metrics(pred, ref)
    diff = pred - ref  # [0,-1,0,-1]
    assert m["rmse"] == np.sqrt(np.mean(diff ** 2))
    assert m["mae"] == np.mean(np.abs(diff))
    assert m["correlation"] > 0.9


def test_metrics_ignore_nan():
    pred = np.array([1.0, 2.0, np.nan, 4.0])
    ref = np.array([1.0, 3.0, 9.0, 4.0])
    m = vmod.compute_metrics(pred, ref)
    assert m["n"] == 3


def test_render_diff_heatmap(tmp_path):
    pred = np.zeros((8, 8), dtype="float32")
    ref = np.ones((8, 8), dtype="float32")
    p = tmp_path / "diff.png"
    vmod.render_diff_heatmap(pred, ref, p)
    assert p.exists() and p.stat().st_size > 0
