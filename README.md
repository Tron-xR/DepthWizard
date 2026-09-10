# DepthWizard

Image-to-3D-terrain pipeline: upload a satellite/aerial photo, get back a real-world-scaled 3D mesh with elevation calibrated against DEM ground truth.

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| **Git** | any recent | clone the repo |
| **uv** | >= 0.4 | Python project manager (installs everything in `server/`) |
| **Python** | 3.10 | pinned by `server/pyproject.toml` |
| **Unity Hub** | 6000.3.11f1 | only needed for the `unity-client/` |

Install uv on Windows: `winget install astral-sh.uv`, or PowerShell: `irm https://astral.sh/uv/install.ps1 | iex`. On macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Clone and Run

```bash
# HTTPS
git clone https://github.com/Tron-xR/DepthWizard.git
# or SSH
git clone git@github.com:Tron-xR/DepthWizard.git
cd DepthWizard
```

### 1. Server (back-end)

```bash
cd server
uv sync                 # creates .venv/ + installs all locked deps (Python 3.10)
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Verify it is up: open `http://127.0.0.1:8000/health` → should return `{"status": "ok"}`.

A fresh clone runs out of the box with the **Depth Anything V2** backend (auto-downloaded
by HF). The heavy pix2pix/IMELE weights and the fine-tuned checkpoint are not in the repo,
so those backends are skipped until you add them (see [Depth Backends](#depth-backends)).

### 2. Unity client (front-end)

1. Open **Unity Hub** → **Add project from disk** → select the `unity-client/` folder.
2. Unity prompts to install **6000.3.11f1** — accept (or open the project exactly as instructed after installing that version).
3. Press **Play**.

The client calls `http://127.0.0.1:8000` by default; change it in the **Settings** screen
if your server runs elsewhere.

### 3. Run the tests

```bash
cd server
uv run pytest -q
```

Expected from a fresh clone: 55 passed, 2 skipped.
The 2 skips are the two opt-in backends whose weights live outside the repo —
install them (below) and the suite becomes 57 passed.

## Architecture

```
unity-client/          Unity 6 (6000.3.11f1) — Android/standalone frontend
  Scripts/
    Core/               ServerManager, DepthWizardApi, Launcher
    UI/                 MainMenu, Upload, Processing, Viewer, Settings screens
    Mesh/               MeshGenerator (runtime heightmap → terrain mesh)
    Camera/             FreeFlyCamera

server/                FastAPI + ML pipeline (Python 3.10, uv)
  app/
    main.py             FastAPI entrypoint
    routes.py           /generate, /status, /download endpoints
    config.py           env-driven config (model paths, DB URL)
    schemas.py          request/response Pydantic models
    db.py               SQLite session helpers
    pipeline/
      depth.py          depth backends: pix2pix GAN (default), Depth Anything V2,
                        IMELE, finetuned DA2 decoder
      calibration.py    least-squares elevation calibration to DEM ground truth
      dem_source.py     OpenTopography / SRTM DEM fetcher + cache
      exporter.py       heightmap PNG + NPZ export
      perlin.py         Perlin noise fallback
      validate.py       input validation
      uploader.py       upload handling
      evaluation.py     held-out site evaluation harness
      decoder_head.py   shared DecoderHead + finetuned DA2 inference

decoder_training/      fine-tuned Depth Anything V2 decoder training
  train_decoder.py      training pipeline (frozen backbone + 3-layer head)
  eval_finetuned.py     held-out evaluation against all 4 backends
```

## Depth Backends

| Backend | Default? | How to enable | Weight |
|---|---|---|---|
| **Depth Anything V2** | yes (fresh clone) | nothing — auto-downloads | HuggingFace, `LiheYoung/depth-anything-small-hf` |
| **pix2pix GAN** | yes (if weight present) | drop a SavedModel in `models/pix2pix` | TF, 512×512 |
| **IMELE** | no | `DEPTHWIZARD_IMELE_MODEL=/path/to/Block0_skip_model_*.tar` | PyTorch (SENet154 + D2/MFF/R) |
| **Finetuned DA2 decoder** | no | `DEPTHWIZARD_FINETUNED_MODEL=/path/to/da2_decoder_finetuned.pt` | PyTorch, 23,873-param head on frozen DA2-small backbone |

Backend dispatch order: finetuned → pix2pix → DA2 → IMELE. The pix2pix GAN only runs when
`models/pix2pix/` exists on disk; otherwise the server uses Depth Anything V2. The IMELE and
finetuned backends are strictly opt-in (env var set), so a fresh clone never breaks.

### Restoring the full model set

The three optional weights are too large for the repo. To replicate:

```bash
# pix2pix GAN (the original default backend)
#   -> acquire the SavedModel and place it at models/pix2pix/

# IMELE building heights
#   -> download Block0_skip_model_*.tar (D2/MFF/R) and export via DEPTHWIZARD_IMELE_MODEL

# Fine-tuned decoder head (train it yourself — see below)
python decoder_training/train_decoder.py --epochs 30 --seed 0 --batch 4
DEPTHWIZARD_FINETUNED_MODEL=decoder_training/checkpoints/da2_decoder_finetuned.pt
```

### Fine-tuning the decoder head

```bash
# 1. Gather training pairs (54 RGB+DEM pairs across hilly/forested/sparse strata)
python gather_training.py            # needs OPENTOPO_API_KEY env var

# 2. Train (frozen DA2 backbone, 3-layer head, 30 epochs)
python decoder_training/train_decoder.py --epochs 30 --seed 0 --batch 4

# 3. Set the env var to use it
DEPTHWIZARD_FINETUNED_MODEL=decoder_training/checkpoints/da2_decoder_finetuned.pt
```

## Tech Stack

- **Server**: Python 3.10, FastAPI, uvicorn, PyTorch 2.5, TensorFlow 2.x, rasterio, scikit-image
- **Client**: Unity 6 (6000.3.11f1), C#, built-in render pipeline
- **DB**: SQLite (dev), production-ready schema
- **Depth models**: pix2pix (GAN), Depth Anything V2 (ViT-small), IMELE, finetuned DA2 decoder head
