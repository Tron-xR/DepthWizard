"""Application configuration and runtime paths."""
from __future__ import annotations

import os
from pathlib import Path

# Project root: server/
SERVER_ROOT: Path = Path(__file__).resolve().parent.parent

# Data dir can be overridden via env (used in tests / clean installs)
DATA_DIR: Path = Path(os.environ.get("DEPTHWIZARD_DATA", SERVER_ROOT / "data"))

CACHE_DIR: Path = DATA_DIR / "cache"
FILES_DIR: Path = DATA_DIR / "files"          # per-job output bundles served over HTTP
DB_PATH: Path = DATA_DIR / "depthwizard.db"

# Server
HOST: str = os.environ.get("DEPTHWIZARD_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("DEPTHWIZARD_PORT", "8000"))

# Pipeline
MAX_WORKING_DIM: int = int(os.environ.get("DEPTHWIZARD_MAX_DIM", "1024"))
# Vertical relief (m) applied to the normalized relative DSM so Unity meshes are legible.
RELATIVE_ELEVATION_RANGE_M: float = float(os.environ.get("DEPTHWIZARD_RELIEF_M", "200"))
# Calibration-health threshold: a prediction whose own relative variation
# (std / |mean|, scale-free) is below this FRACTION is treated as effectively
# constant (warned as "prediction_near_constant") even when the affine fit is
# numerically valid. Comparing against the prediction's own magnitude (not the
# reference's std) keeps 0..1-normalized relative-depth maps from being flagged
# purely because the reference is in meters.
MIN_RELATIVE_STD: float = max(
    float(os.environ.get("DEPTHWIZARD_MIN_RELATIVE_STD", "0.05")), 1e-9
)
# Calibration inlier-support threshold: a RANSAC fit supported by fewer than
# this FRACTION of its training pixels describes a small sub-population
# (extrapolation, not calibration), warned as "too_few_inliers" by
# calibration.fit_affine. Documented here (not inline) so the API, exports,
# and the benchmark harness all read one value.
# ponytail: fixed 10% heuristic. Upgrade path: keep the supported fraction on
# the fit and let the UI render support confidence instead of a warning flag.
CALIB_MIN_INLIER_FRACTION: float = float(
    os.environ.get("DEPTHWIZARD_CALIB_MIN_INLIER_FRACTION", "0.10")
)
# Depth backbone. The pix2pix GAN SavedModel is the default backend when present
# (models/pix2pix in the repo root); set DEPTHWIZARD_TF_MODEL to a different
# SavedModel, or set it to an empty string to force the transformers/HF
# depth-anything fallback instead.
_DEFAULT_TF_MODEL: Path = Path(__file__).resolve().parents[2] / "models" / "pix2pix"
TF_MODEL_PATH: str = os.environ.get("DEPTHWIZARD_TF_MODEL", "")
if not TF_MODEL_PATH and _DEFAULT_TF_MODEL.exists():
    TF_MODEL_PATH = str(_DEFAULT_TF_MODEL)
DEFAULT_DEPTH_MODEL: str = os.environ.get(
    "DEPTHWIZARD_DEPTH_MODEL", "LiheYoung/depth-anything-small-hf"
)
# IMELE building-height backbone (SENet154 + D2/MFF/R, PyTorch). Strictly
# opt-in: set DEPTHWIZARD_IMELE_MODEL to a Block0_skip_model_*.tar path. It is
# not auto-discovered, so existing switches stay untouched: empty TF_MODEL_PATH
# still means HF depth-anything, and the pix2pix default is unchanged.
IMELE_MODEL_PATH: str = os.environ.get("DEPTHWIZARD_IMELE_MODEL", "")
# Fourth backend: fine-tuned DA2-small + trained decoder head checkpoint
# (produced by decoder_training/train_decoder.py). Opt-in only.
FINETUNED_MODEL_PATH: str = os.environ.get("DEPTHWIZARD_FINETUNED_MODEL", "")
# DA3 (Depth Anything 3) backbones: strictly opt-in, one env var per size
# (value = HF repo id or local snapshot dir; empty = disabled). Setting more
# than one is an error (depth._selected_da3), so "which model ran" is never
# ambiguous. Licences: SMALL/BASE/MONO-LARGE are Apache-2.0; LARGE/GIANT are
# CC BY-NC 4.0 (non-commercial use only).
DA3_SMALL_MODEL: str = os.environ.get("DEPTHWIZARD_DA3_SMALL_MODEL", "")
DA3_BASE_MODEL: str = os.environ.get("DEPTHWIZARD_DA3_BASE_MODEL", "")
DA3MONO_LARGE_MODEL: str = os.environ.get("DEPTHWIZARD_DA3MONO_LARGE_MODEL", "")
DA3_LARGE_MODEL: str = os.environ.get("DEPTHWIZARD_DA3_LARGE_MODEL", "")
DA3_GIANT_MODEL: str = os.environ.get("DEPTHWIZARD_DA3_GIANT_MODEL", "")

# Optional prediction-artifact export root. Artifacts are written ONLY when a
# request opts in (e.g. /evaluate?save_artifacts=true), under
# <OUTPUT_DIR>/<backend>/<input-stem>/ so pix2pix vs IMELE runs never collide.
# Overridable via DEPTHWIZARD_OUTPUT_DIR; defaults to server/outputs.
OUTPUT_DIR: Path = Path(os.environ.get("DEPTHWIZARD_OUTPUT_DIR", SERVER_ROOT / "outputs"))

# Reference DEM fetch
DEM_FETCH_TIMEOUT_S: float = float(os.environ.get("DEPTHWIZARD_DEM_TIMEOUT", "30"))
# Optional OpenTopography API key (free). If absent, fetch may fall back.
OPENTOPO_API_KEY: str = os.environ.get("DEPTHWIZARD_OPENTOPO_API_KEY", "")
# Max reference tile width in meters after one OpenTopography call (kept modest)
DEM_MAX_TILE_M: float = float(os.environ.get("DEPTHWIZARD_DEM_TILE_M", "5000"))


def ensure_dirs() -> None:
    """Create all runtime directories if missing."""
    for p in (CACHE_DIR, FILES_DIR):
        p.mkdir(parents=True, exist_ok=True)
