# Phase 2 design: DA3 backbones on the frozen test split

Branch `backbone-eval` (from tag `phase1b-done` @ `f7993ca`). Adds Depth Anything 3
(DA3) monocular-depth backbones as strictly opt-in alternatives, scores them on
the frozen Phase 1b TEST split with the existing harness, and adds two trivial
control arms. Nothing about the frozen evaluation is changed: manifest,
`acceptance.yaml`, splits, seeds and thresholds are untouched; the DA3 backends
are disabled unless a `DEPTHWIZARD_DA3_*` env var is set, so default dispatch is
byte-for-byte the Phase 1b behavior.

## Backends

Upstream: github.com/ByteDance-Seed/Depth-Anything-3 @ `3d835ec` (`pip install
--no-deps` from a clone; see `experiments/backbone_eval/requirements-freeze.txt`).
HF org is `depth-anything` (not ByteDance-Seed).

| slug | HF id | params | licence | env var |
|---|---|---|---|---|
| `da3-small` | `depth-anything/DA3-SMALL` | 34 M | Apache-2.0 | `DEPTHWIZARD_DA3_SMALL_MODEL` |
| `da3-base` | `depth-anything/DA3-BASE` | 0.12 B | Apache-2.0 | `DEPTHWIZARD_DA3_BASE_MODEL` |
| `da3mono-large` | `depth-anything/DA3MONO-LARGE` | 0.35 B | Apache-2.0 | `DEPTHWIZARD_DA3MONO_LARGE_MODEL` |
| `da3-large` | `depth-anything/DA3-LARGE-1.1` | 0.35 B | CC BY-NC 4.0 | `DEPTHWIZARD_DA3_LARGE_MODEL` |
| `da3-giant` | `depth-anything/DA3-GIANT-1.1` | 1.15 B | CC BY-NC 4.0 | `DEPTHWIZARD_DA3_GIANT_MODEL` |

Env values are an HF repo id or a local snapshot dir; empty = disabled.
Dispatch order: finetuned > imele > DA3 > TF/pix2pix > HF DA2. Setting more than
one `DEPTHWIZARD_DA3_*` var raises `DepthModelError` (five silently-mixable
candidates make "which model ran" unanswerable from artifacts). `da3-large` and
`da3-giant` are **non-commercial** (CC BY-NC 4.0) and are flagged as such in the
report; `da3-giant` (1.15 B params ≈ 4.6 GB fp32) does not fit this box's 4 GB
RTX 3050 and is marked hardware-skipped, not run.

## Sign convention (fixed, documented rule)

- The DA3 API (`DepthAnything3.inference`) returns `Prediction.depth`, which is
  z-depth: **larger = farther from the camera**. Evidence from the upstream
  code: the GLB exporter unprojects `point = depth * ray(z=1)` (planar distance
  divided by the ray's z-component), and the metric helper scales `depth *
  focal / 300` for metric recovery. The model card likewise states mono depth is
  predicted "directly" (not disparity) unlike DA2.
- The harness convention is defined by the established DA2/HF adapter in
  `app/pipeline/depth.py`: DA2's raw tensor is inverse depth (nearer = larger),
  so it is inverted (`invert_depth=True`), i.e. rDSM **increases with
  camera-to-ground distance** (brighter = higher in the harness's reading).
  pix2pix / IMELE / finetuned were trained to "brighter = higher" and skip the
  flip.
- Rule for all five DA3 models: `invert_depth = False` — a z-depth map already
  increases with distance. One boolean per backend family, declared in code,
  decided from the model's output definition; never chosen per tile, per site or
  per result.
- Inversion visibility: the harness now reports `raw_r_opposite = -raw_r` next
  to `raw_r` (per-tile CSV, aggregates, receive and markdown tables,
  bootstrap CI). For nadir imagery "farther" physically reads as *lower* terrain
  (higher ground is closer to the camera), which is exactly why the opposite
  convention is reported alongside instead of being silently selected. A family-
  level sign mistake is one boolean in the adapter, and both signs remain
  visible in the artifacts.
- Resolution: DA3 returns depth on its processed grid (long side 504 default);
  `infer_relative_dsm` resamples to the tile grid with LANCZOS, the same path
  the DA2/HF arm already uses. `process_res` stays at the library default — no
  tuning on the test split.

## Harness integration (reuse, not fork)

- Prediction generation: `tools/benchmark_tiles.py run --protocol historical
  --backend da3-*|brightness|sobel --tiles <26 test ids>` writes a run dir with
  `predictions/<tile>_relative.npy`, identity sidecars, `model_info.json`,
  `run.json`, `environment.json` — the exact artifact format the frozen Phase 1b
  arms use.
- Scoring: `python -m app.pipeline.evaluation benchmark --split test` — the same
  code path and thresholds as Phase 1b, with the new run dirs passed via
  `--predictions NAME=DIR`. `_DEFAULT_PREDICTION_RUNS` is untouched.
- Controls: `brightness` (channel-mean grayscale) and `sobel` (gradient
  magnitude of grayscale) — trivial non-depth predictors that bound how much of
  the score raw intensity structure alone explains. No sign rule claimed; the
  `raw_r_opposite` column shows the flipped view.
- Per-model runtime metadata: one-time model load seconds (`run.json:
  model_load_seconds`, measured in the tool) and process GPU peak
  (`environment.json: cuda_max_memory_allocated_mb`).
- Provenance per DA3 run: `model_info.json` gains `hf_repo_id`, `hf_revision`
  (hub commit sha) and the sha256 of the weight file
  (`depth.da3_checkpoint_provenance`).

## Environment

Separate venv `experiments/backbone_eval/.venv` (Python 3.10.20, torch
2.5.1+cu124 — matching the production stack) with the frozen dependency list in
`experiments/backbone_eval/requirements-freeze.txt`. The production venv and
`server/pyproject.toml` are not modified. DA3 is installed `--no-deps` plus an
explicit dependency list (numpy<2, addict, omegaconf, einops, safetensors,
opencv-python, imageio, trimesh, moviepy==1.0.3, matplotlib, plyfile, evo,
pycolmap, gsplat — the import chain of `depth_anything_3.api` eagerly imports
the export/visualization modules).

## Evaluation protocol

- Frozen TEST split only (26 tiles / 5 sites / 3 strata), from
  `server/data/manifest.csv` (read-only).
- Sanity gate first: re-run the benchmark with the default frozen arms and
  compare against `docs/benchmarks/phase1b/benchmark_test.md` (da2 cal RMSE
  154.105, raw r -0.0548882, etc.). Any drift blocks the new arms from being
  trusted.
- Then the canonical run includes all arms at once (default 4 + DA3 + 2
  controls) so every comparison shares one mask/tile population per tile.
- Report per model and per stratum: raw signed r (and opposite), skill vs const,
  calibrated RMSE, status counts, site-weighted aggregation with bootstrap CIs;
  plus runtime/VRAM per model and the licence flags. No tuning on test.

## Out of scope / non-goals

- No production default change, no new dependency in `server/pyproject.toml`.
- `data/`, `results/`, `server/config/acceptance.yaml`, manifest and
  downloader/gather scripts are read-only for this phase.
- No DA3 fine-tuning, no resolution sweep, no multi-view inference (single
  ortho tile per prediction, one image per `inference()` call).
