# DepthWizard

Image-to-3D-terrain pipeline: upload a satellite/aerial photo, get back a real-world-scaled 3D mesh with elevation calibrated against DEM ground truth.

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

sample_images/         test satellite/aerial photos
```

## Quick Start

### Server

```bash
cd server
uv sync                  # install dependencies
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check: `http://127.0.0.1:8000/health`

### Unity Client

Open `unity-client/` in Unity 6 (6000.3.11f1). Press Play.

The client talks to `http://127.0.0.1:8000` by default — configure in the Settings screen.

### Tests

```bash
cd server
uv run pytest -q
```

## Depth Backends

| Backend | Env var to disable | Weight |
|---|---|---|
| **pix2pix GAN** (default) | `DEPTHWIZARD_TF_MODEL=""` | TF, 512×512 |
| **Depth Anything V2** | `DEPTHWIZARD_DA2=""` | HuggingFace, `LiheYoung/depth-anything-small-hf` |
| **IMELE** | `DEPTHWIZARD_IMELE=""` | TF, external download |
| **Finetuned DA2 decoder** | (enable with `DEPTHWIZARD_FINETUNED_MODEL=path/to/checkpoint.pt`) | PyTorch, 23,873-param head on frozen DA2-small backbone |

Set `DEPTHWIZARD_FINETUNED_MODEL` to a checkpoint path to enable the finetuned backend. The server picks backends in this order: finetuned → pix2pix → DA2 → IMELE.

### Fine-tuning the decoder head

```bash
# 1. Gather training pairs (54 RGB+DEM pairs across hilly/forested/sparse strata)
python gather_training.py          # needs OpenTopography API key in env

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
