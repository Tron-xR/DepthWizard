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
