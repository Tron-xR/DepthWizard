# Remote-sensing height backends: design (Phase 3 candidatures)

Step 1 findings + registration plan for two off-the-shelf remote-sensing height
models, evaluated as opt-in backends on this repo's frozen benchmark splits.
Nothing here is auto-loaded; the default dispatch is unchanged.

## Models

### RS3DAda (SynRS3D, NeurIPS 2024)
- Weights: `JTRNEO/RS3DAda` → `RS3DAda_vitl_DPT_height.pth` (1.47 GB, 366.5 M
  params, fp32; Zenodo 13940407 mirror). Code: github.com/JTRNEO/SynRS3D @
  `ab5a485`.
- Architecture: DINOv2 ViT-L/14 encoder + DPT decoder, two heads
  (`regression` nclass=1, `segmentation` nclass=8). We use only the regression
  head.
- Inference API (upstream `infer_height.py`): construct `DPT_DINOv2(encoder=
  "vitl", head_configs=[{"name":"regression","nclass":1},{"name":
  "segmentation","nclass":8}], pretrained=False)`, `load_state_dict(torch.load(
  ckpt))`, `.eval().cuda()`; input = RGB uint8 0-255 normalised with ImageNet
  mean (123.675, 116.28, 103.53) / std (58.395, 57.12, 57.375) at raw scale;
  output = `outputs["regression"]`, a raw linear map, **nDSM in metres** (no
  activation, no normalisation, no clip in the export path).
- Input size / GSD: trained on SynRS3D at GSD 0.05-1.0 m (three bins); upstream
  demo runs 1022 px patches (multiple of 14), stride 511, optional 4-flip TTA.
  No single GSD is assumed; the model is fully convolutional and GSD-agnostic
  in code, but its accuracy census was measured on 0.05-1 m imagery. The
  frozen Swiss tiles are 1 m/px - inside that census - so RS3DAda runs them
  **native, no resample** (`gsd_model=None`).
- Licences: code **MIT** (Jian Song, 2024); weights **MIT** (HF card + Zenodo
  metadata agree). Training data: synthetic SynRS3D — **licence discrepancy to
  flag: HF dataset card says CC BY-NC 4.0, Zenodo record 13905264 says CC BY
  4.0**. RS3DAda training also involved unlabeled real data (DFC18/19, GeoNRW,
  OGC — CC BY 4.0) via UDA (no labels consumed).
- VRAM: not documented upstream. On this box (RTX 3050 4 GB) 518 px bf16
  windows fit (altimap author ran the same architecture at 518 px on an RTX
  3050 laptop); 1022 px fp32 is not viable in 4 GB, so we run 518/259 instead
  (documented deviation, reported in the results table).

### altimap-height v1 / v2 (`Dilavesh/altimap-height`)
- One HF repo, no revision tags: **v1** = `best.pth` (root), **v2** =
  `v2/best.pth`; 1.47 GB each, same DPT_DINOv2 viT-L architecture as RS3DAda.
- Inference API: load exactly like RS3DAda (same class from the SynRS3D repo);
  windowed slider at **518 px, half-stride, feathered blending, bf16 autocast**;
  output = **nDSM metres, clamped at >= 0** ("height above ground is
  non-negative by definition"); no scaling/post-hoc correction. Same ImageNet
  normalisation constants on RGB uint8.
- Input size / GSD: **documented 0.33 m/px target** (GAMUS); v2 training adds
  1.3-6x coarser degraded inputs (eval spans 0.35-10 m). Usage comment: "input
  at ~0.33 m/px". Pipeline: a **1 m Swiss tile is bilinearly upsampled 3.03x
  to 0.33 m/px before windowing, and the metre map is area-averaged back to
  the 1 m input grid** once inference is done (values stay metres on both
  grids; see "GSD handling" below). The card's own NAIP table shows object
  heights "fade to flat by 5-10 m" pixels, so native-GSD inference on 1 m
  tiles is deliberately not the default.
- Licences: repo/weights **MIT**; v1 trained on GAMUS (**CC BY 4.0**); **v2
  additionally on SynRS3D → the card itself says "treat v2 as
  non-commercial"** (NC propagates from the dataset licence). v2 outputs must
  carry the NC flag.
- Reported accuracy: GAMUS test RMSE v1 5.02 m / v2 4.55 m (zero-shot RS3DAda
  7.25 m); building RMSE 8.28 / 6.74 m; forest bias ~ -10.8 m out of domain.
- VRAM: upstream SETUP.md says **6 GB VRAM or more** (24 GB to train). This box
  has 4 GB — bf16 at 518 px windows is the mitigation; smoke test measures the
  actual peak and reports it. If it OOMs, the arm is hardware-skipped and
  reported (same policy as da3-giant in the backbone eval).

## Registration (smallest diff, mirrors the DA3 opt-in pattern)

- `config.py`: `DEPTHWIZARD_RS3DADA_MODEL`, `DEPTHWIZARD_ALTIMAP_V1_MODEL`,
  `DEPTHWIZARD_ALTIMAP_V2_MODEL` (empty = off). Value = HF repo id, a local
  `.pth` path, or a directory containing the upstream filename.
- `depth.py`: `RS_BACKENDS` table `(slug, config attr, repo id, filename)`;
  `_selected_rs()` (single-select, same as `_selected_da3`); `_RsHeightBackend`
  (`output_kind = "ndsm_metres"`); `infer_ndsm_metres(rgb)` returns the raw
  metre map (no normalisation, no sign flip, no calibration) and raises for
  relative-depth backends; dispatch order finetuned > imele > rs > da3 > TF with
  a guard against configuring an rs and a da3 backend at once;
  `rs_checkpoint_provenance()` records hub revision + weight sha256 like
  `da3_checkpoint_provenance`.
- Model code: vendor the SynRS3D `models/dpt.py` + `models/blocks.py` (MIT,
  pinned `ab5a485`) into `server/app/pipeline/rs_dpt.py` with the licence
  header + modification note; `server/app/pipeline/rs_height.py` is our
  adapter (checkpoint load, cached DINOv2 torch.hub build, 518 px windowed
  metre prediction). No new dependency: torch + torch.hub (DINOv2 code is
  fetched once into the torch cache, not a pip package) + huggingface_hub,
  all already in the server lock.
- `benchmark_tiles.py` / evaluation: metric backends save `predictions/
  <tile>_ndsm.npy` (float32 metres) + the usual sidecars; the rDSM scoring path
  is untouched.

## Evaluation modes (frozen TEST split, same valid mask)

- **ndsm**: reference = Swiss nDSM raster; per tile, shared valid mask
  (finite ref + finite preds); arms = each backend's predicted nDSM. Metric
  backends are scored raw AND with the harness's spatial-block affine fit as a
  drift diagnostic (scale != 1 means the checkpoint is off on this domain);
  relative backends (da2/da3) and the sobel/brightness controls are scored via
  the SAME calibrated arm (they have no metric scale without it). Baselines:
  `predict_zero` (zeros) and `const_mean` (calibration-region reference mean —
  no eval leakage). Report per arm: RMSE, MAE, bias, raw signed r (both signs),
  skill vs predict-zero, per stratum (urban/sparse/hilly/forest),
  site-weighted, bootstrap CIs over sites; failed tiles stay in counts.
- **dsm**: reference = Swiss DSM; arm prediction = prior + predicted nDSM
  (prior = per-tile prior raster from the manifest; **nothing is subtracted
  from the prior**). Double-counting check: same metrics for the ORACLE
  (prior + true nDSM) and for prior-alone. If an arm's RMSE approaches
  prior-alone, its "nDSM" is re-explaining terrain the prior already carries.
- Interface to be finalised when Phase 2 lands: a `--height-manifest` CSV with
  `tile_id,ndsm_path,dsm_path,prior_path` (absolute or repo-relative paths);
  the harness adapts to the actual Swiss columns at final-run time.
- No tuning on the test split; `acceptance.yaml` and the manifest are
  read-only; the reference metric for acceptance stays the frozen Phase 1b
  rDSM protocol - the two new modes are diagnostic evaluations of headline
  question "can an RS height model add anything over the prior on Swiss data".

## GSD handling (1 m Swiss tiles -> checkpoint target)

- `rs_height.predict_height(gsd_in, gsd_model)` resamples only when both are
  known: RGB bilinear to `gsd_model` (1 m / 0.33 = 3.03x for altimap; the
  171 px smoke tile lands exactly on one 518 px window), then the windowed
  metre map back to the input grid - `area` when downscaling, `bilinear` when
  upscaling. Resampling never rescales values: the output is nDSM metres on
  whatever grid is returned (constant-field roundtrip covered by a test).
- Per-checkpoint target: altimap `gsd_model=0.33`; RS3DAda `None` (1 m is
  inside its 0.05-1.0 m training census). `benchmark_tiles._default_infer`
  passes `gsd_in=1.0` for the frozen Swiss tiles (`_TILE_GSD_M`).
- altimap_v1's expected output range, from the model card (GAMUS test): nDSM
  metres >= 0 (clamped), typical RMSE 5.02 m, tallest real objects 151 m LiDAR
  / 66-104 m predicted. The synthetic-urban smoke asserts [0, 200) m and a
  building signal > 1 m - a broken backend reads flat or absurd, not subtle.

## Compute device, VRAM cap, backbone pin (hardening before the final run)

- Device is explicit: `DEPTHWIZARD_RS_DEVICE` (argument > env > auto). Auto
  means `torch.cuda.device_count() > 0`, **never**
  `torch.cuda.is_available()` - the latter was observed returning True with
  zero visible devices (torch 2.5.1 / Windows / empty CUDA_VISIBLE_DEVICES)
  and crashed inside autocast. An explicit cuda request with 0 devices is an
  error naming the env var, not a silent CPU fallback.
- Provenance (run `model_info` -> `compute`): chosen device, its source
  (explicit/env/auto), CUDA device count, and the applied
  `DEPTHWIZARD_CUDA_MEM_FRACTION` (None when uncapped or CPU).
- DINOv2 backbone code is pinned, not the mutable `main` ref: the worktree
  `.torch-cache/hub` copy is loaded first (no network once cached), and
  `model_info` -> `backbone_dinov2` records the zipball's commit sha +
  sha256. Falls back to TORCH_HOME, then a download, only when nothing is
  cached.

## Chosen defaults (reported as decisions)

- Window 518 / stride 259 / feather / bf16 for all three rs checkpoints
  (4 GB VRAM constraint; altimap's own 3050 usage). TTA off by default.
- Heights clamped >= 0 for altimap (its documented behaviour); RS3DAda left
  unclamped (its export path has no clip) — both recorded.
- altimap_v2 is registered but every output (model_info, limitations,
  report) carries the non-commercial flag.
