"""Prediction-artifact export: exact-float npy/png/tif/metrics.json.

Spec coverage (no models required - all arrays are synthetic):
  - npy preserves exact float values/dtypes (never uint8, never re-normalized)
  - saved npy matches the actual prediction array
  - PNG is visualization-only (robust stretch on a COPY) and metrics are
    byte-for-byte unaffected by exporting
  - GeoTIFF preserves CRS + transform for georeferenced input; float values
  - output dims match the final prediction
  - metrics.json carries the required stats (shape/dtype/min/max/mean/std,
    p1..p99, unique, the reported metrics, calibration health, backend/model)
  - both pix2pix and IMELE use the SAME code path (one function, per-backend
    subdirectories) - no duplicate inference
"""
import json

import numpy as np
import pytest

from app.pipeline import artifacts
from app.pipeline import depth


def _synth_prediction(seed=0, shape=(32, 24), calibration="valid"):
    rng = np.random.default_rng(seed)
    # raw model output: pre-normalization, arbitrary range (e.g. GAN tanh-ish)
    raw = (rng.uniform(-0.8, 0.3, shape)).astype("float32")
    relative = ((raw - raw.min()) / (raw.max() - raw.min())).astype("float32")
    if calibration == "valid":
        calibrated = (120.0 + 400.0 * relative).astype("float64")
    else:
        calibrated = None
    return relative, calibrated, raw


def test_npy_preserves_exact_float_values(tmp_path):
    relative, calibrated, raw = _synth_prediction()
    out = artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.png", relative=relative, calibrated=calibrated,
        metrics={}, crs=None, transform=None, bounds=None)
    arr = np.load(tmp_path / "pix2pix" / "scene" / "predicted_depth.npy")
    assert arr.dtype == calibrated.dtype  # float64 kept, not uint8/32
    assert np.array_equal(arr, calibrated)
    assert arr.min() > 100.0  # raw calibration scale intact - not 0..1


def test_saved_npy_matches_prediction(tmp_path):
    relative, calibrated, raw = _synth_prediction()
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.png", relative=relative, calibrated=calibrated,
        raw=raw,
        metrics={}, crs=None, transform=None, bounds=None)
    for name in ("predicted_depth.npy", "raw_prediction", "relative_prediction", "calibrated_prediction"):
        fname = name if name.endswith(".npy") else f"{name}.npy"
        saved = np.load(tmp_path / "pix2pix" / "scene" / fname)
        expected = {"predicted_depth.npy": calibrated, "calibrated_prediction": calibrated,
                    "relative_prediction": relative, "raw_prediction": raw}[name]
        assert saved.dtype == expected.dtype
        assert np.array_equal(saved, expected)


def test_png_visualization_only_and_metrics_untouched(tmp_path):
    metrics = {"mae": 9.37, "rmse": 12.75, "correlation": 0.4878,
               "correlation_reason": None, "calibration_status": "warning"}
    relative, calibrated, raw = _synth_prediction()
    pred_copy = calibrated.copy()
    out = artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.png", relative=relative, calibrated=calibrated,
        raw=raw,
        metrics=metrics, crs=None, transform=None, bounds=None)

    base = tmp_path / "pix2pix" / "scene"
    assert (base / "predicted_depth.png").is_file()

    # the export must not mutate the caller's prediction array
    assert np.array_equal(calibrated, pred_copy)

    # PNG is 8-bit grayscale, and metrics.json reports the SAME metrics
    from PIL import Image
    img = Image.open(base / "predicted_depth.png")
    assert img.mode == "L"
    meta = json.loads((base / "metrics.json").read_text())
    for k, v in metrics.items():
        if v is not None or k != "correlation_reason":
            assert k in meta["metrics"]
    assert meta["metrics"]["mae"] == pytest.approx(metrics["mae"])
    assert meta["metrics"]["rmse"] == pytest.approx(metrics["rmse"])
    assert meta["metrics"]["correlation"] == pytest.approx(metrics["correlation"])
    # viz params are recorded (min/max/percentiles used)
    assert isinstance(meta["visualization"]["min"], float)
    assert meta["visualization"]["percentiles"] == [1.0, 99.0]


def test_geotiff_preserves_crs_transform_for_georeferenced_input(tmp_path):
    from rasterio.transform import from_origin

    relative, calibrated, raw = _synth_prediction()
    transform = from_origin(78.0, 30.55, 1.0 / 3600.0, 1.0 / 3600.0)
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.tif", relative=relative, calibrated=calibrated,
        raw=raw,
        metrics={}, crs="EPSG:4326", transform=transform,
        bounds=[78.0, 30.4, 78.01, 30.55])

    import rasterio
    with rasterio.open(tmp_path / "pix2pix" / "scene" / "predicted_depth.tif") as dst:
        assert dst.crs == rasterio.crs.CRS.from_epsg(4326)
        assert dst.transform == transform
        assert dst.count == 1
        assert dst.height == calibrated.shape[0]
        assert dst.width == calibrated.shape[1]
        assert dst.dtypes == ("float32",)  # float, never uint8
        written = dst.read(1)
    assert written.dtype == np.float32
    assert np.allclose(written, calibrated.astype("float32"), equal_nan=True)


def test_output_dims_match_final_prediction(tmp_path):
    relative, calibrated, raw = _synth_prediction(shape=(48, 36))
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.png", relative=relative, calibrated=calibrated,
        raw=raw,
        metrics={}, crs=None, transform=None, bounds=None)
    base = tmp_path / "pix2pix" / "scene"
    assert np.load(base / "predicted_depth.npy").shape == calibrated.shape
    assert np.load(base / "relative_prediction.npy").shape == relative.shape
    assert np.load(base / "calibrated_prediction.npy").shape == calibrated.shape
    from PIL import Image
    assert Image.open(base / "predicted_depth.png").size[::-1] == calibrated.shape


def test_metrics_json_required_stats(tmp_path):
    relative, calibrated, raw = _synth_prediction()
    metrics = {"mae": 5.0, "rmse": 7.0, "correlation": 0.1,
               "calibration_status": "valid", "calibration_warning": None,
               "calibration_scale": 3.0, "calibration_offset": 10.0}
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="some-path",
        input_filename="scene.tif", relative=relative, calibrated=calibrated,
        raw=raw,
        ground_truth=calibrated + np.random.default_rng(1).normal(0, 5, calibrated.shape),
        metrics=metrics, crs="EPSG:4326", transform=None, bounds=None)
    meta = json.loads((tmp_path / "pix2pix" / "scene" / "metrics.json").read_text())
    pred = meta["prediction"]
    for key in ("shape", "dtype", "min", "max", "mean", "std", "unique"):
        assert key in pred, key
    for p in (1, 5, 25, 50, 75, 95, 99):
        assert f"p{p}" in pred, f"p{p}"
    assert meta["georeferenced"] is False  # crs present but no transform
    assert meta["crs"] == "EPSG:4326"
    assert meta["input_filename"] == "scene.tif"
    assert meta["backend"] == "pix2pix"
    assert meta["model"] == "some-path"
    assert meta["prediction_dimensions"] == list(calibrated.shape)
    assert meta["ground_truth"]["mean"] is not None
    assert meta["metrics"]["mae"] == metrics["mae"]
    assert meta["metrics"]["calibration_offset"] == metrics["calibration_offset"]


def test_export_does_not_change_reported_metrics(tmp_path):
    relative, calibrated, raw = _synth_prediction()
    before = {"mae": 9.397, "rmse": 12.746, "correlation": 0.489,
              "calibration_status": "warning", "calibration_warning": "negative_scale",
              "calibration_scale": -2.9, "calibration_offset": 10.0}
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="imele", model_identifier="imele.tar",
        input_filename="scene.png", relative=relative, calibrated=calibrated,
        raw=raw,
        metrics=before, crs=None, transform=None, bounds=None)
    meta = json.loads((tmp_path / "imele" / "scene" / "metrics.json").read_text())
    for k, v in before.items():
        assert meta["metrics"][k] == v or (v is None and k in meta["metrics"])


def test_both_backends_share_one_artifact_path(tmp_path):
    # Same function, per-backend subdirs (outputs/pix2pix vs outputs/imele),
    # no duplicate inference: only ONE export call per backend, raw passed in.
    relative, calibrated, raw = _synth_prediction()
    for backend in ("pix2pix", "imele"):
        artifacts.export_prediction_artifacts(
            out_root=tmp_path, backend=backend, model_identifier=backend,
            input_filename="scene.png", relative=relative, calibrated=calibrated,
            raw=raw,
            metrics={}, crs=None, transform=None, bounds=None)
        assert (tmp_path / backend / "scene" / "predicted_depth.npy").is_file()
        assert (tmp_path / backend / "scene" / "metrics.json").is_file()


def test_backend_slug_maps_loaded_model_classes():
    class _Fake:  # mimics a stub object; we assert via class name mapping
        def __init__(self, name):
            self.__class__ = type(name, (), {})

    depth._model_cache["model"] = _Fake("_TfSavedModel")
    try:
        assert depth.backend_slug() == "pix2pix"
        depth._model_cache["model"] = _Fake("_ImeleModelBackend")
        assert depth.backend_slug() == "imele"
        depth._model_cache["model"] = _Fake("_FinetunedDepthBackend")
        assert depth.backend_slug() == "finetuned"
        depth._model_cache["model"] = _Fake("SomethingElse")
        assert depth.backend_slug() == "depth_anything"
    finally:
        depth._model_cache.pop("model", None)


def test_non_georeferenced_input_writes_plain_tif(tmp_path):
    relative, calibrated, raw = _synth_prediction()
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.png", relative=relative, calibrated=calibrated,
        raw=raw,
        metrics={}, crs=None, transform=None, bounds=None)
    import rasterio
    with rasterio.open(tmp_path / "pix2pix" / "scene" / "predicted_depth.tif") as dst:
        assert dst.crs is None
        assert dst.height == calibrated.shape[0]


def test_same_filename_two_jobs_never_share_artifact_dir(tmp_path):
    # Terrain A then Terrain B, both uploaded as "scene.png" in one session.
    # Job-scoped export must keep them in separate directories so B can never
    # serve A's already-written artifacts, even though the stem is identical.
    rel_a, cal_a, _ = _synth_prediction(seed=11)
    rel_b, cal_b, _ = _synth_prediction(seed=22)
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.png", job_id="job-A",
        relative=rel_a, calibrated=cal_a, raw=None,
        metrics={}, crs=None, transform=None, bounds=None)
    artifacts.export_prediction_artifacts(
        out_root=tmp_path, backend="pix2pix", model_identifier="m",
        input_filename="scene.png", job_id="job-B",
        relative=rel_b, calibrated=cal_b, raw=None,
        metrics={}, crs=None, transform=None, bounds=None)
    dir_a = tmp_path / "pix2pix" / "job-A"
    dir_b = tmp_path / "pix2pix" / "job-B"
    assert dir_a != dir_b
    # stem fallback dir is NOT created when job_id is threaded through
    assert not (tmp_path / "pix2pix" / "scene").exists()
    # each job's dir holds its own prediction, not the sibling's
    assert np.array_equal(np.load(dir_a / "predicted_depth.npy"), cal_a)
    assert np.array_equal(np.load(dir_b / "predicted_depth.npy"), cal_b)