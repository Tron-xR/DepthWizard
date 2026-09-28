# Task 2 — RGB vs coarse-prior experiment

Determines whether RGB imagery adds elevation information beyond a coarse DEM prior.

## Verdict
**The image adds essentially nothing usable.** Model arms (B = RGB+coarse, D = coarse-only)
are 3-8x worse in absolute error than a free bicubic upsampler of the coarse DEM (C), and
the image's gain never survives the absolute-error comparison. See `../../18-rgb-vs-coarse-prior.md`.

## Layout
- `data/metrics/` — per-tile metrics, one file per arm-seed (`metrics_*`, truth = COP30/Tilezen) and per-arm-seed vs independent USGS 3DEP (`metrics3dep_*`, 56 CONUS tiles).
- `data/curves/` — training curves (120 epochs = 4 folds x 30), one per arm-seed.
- `data/*.json` — locked folds, per-tile bbox/pitch metadata, and the 3DEP fetch manifest.
- `scripts/` — analysis + fetch scripts used to produce the report.

## Reproducing
Scripts were executed from a scratch workspace (`%LOCALAPPDATA%\Temp\opencode\exp`) and
embed absolute Windows paths; checkpoints (4 folds x 3 seeds per arm), the 463 MB
`coarse_elev/*.npy`, and the 127 MB `dem3dep/*.tif` are intentionally not committed.
To rerun, adjust `EXP` / `ROOT` in each script, add the `server/` tree to `PYTHONPATH`
(they import studio pipeline modules and use `.venv\Scripts\python.exe`), and drop the
train/heldout/geo_test DEM dirs in place.

Pipeline (in order): `prepare_coarse_elev.py` (block-mean + bicubic-up 90/250 m per tile)
-> `metrics_lib.py` (shared scorer) -> `exp_run_metrics.py` (affine calib, native or
`--truth dem3dep` truth) -> `exp_agg_report.py` (tables + paired block bootstrap + verdict).
Weights/checkpoints for arms B/D were trained with the repo's standard pix2pix training path.

## Full report

# Task 2 report: does RGB add information beyond a coarse DEM prior?

## Setup
- Ring-2 spatial-block CV, 4 folds (locked), 3 seeds per model arm.
- Arms: C = bicubic-upsampled coarse (unit slope, no model); B = DA2-small + RGB + coarse; D = same architecture, RGB zeroed.
- Per tile, deterministic 80/20 pixel split; affine calibration (pred->meters) fit on the 80% for model arms; metrics on the 20%.
- resid_r = corr(pred_elev - coarse_up, truth - coarse_up): detail above the coarse prior. C is 0 by construction.
- Extra tiles (held-out + geo_test) are never in training; CV bootstrap block id = (ci//2, cj//2) of the 2x2 lattice.

## All tiles (mean of seeds for model arms)
| arm | mae (m) | rmse (m) | raw_r | resid_r | raw_r>0 |
|---|---|---|---|---|---|
| C90 | 5.8 | 8.7 | +0.998 | +0.000 | 64/64 |
| B90 | 68.8 | 90.5 | +0.854 | -0.020 | 64/64 |
| D90 | 98.1 | 123.7 | +0.483 | -0.038 | 64/64 |
| C250 | 16.6 | 24.0 | +0.988 | +0.000 | 64/64 |
| B250 | 74.8 | 96.3 | +0.807 | +0.047 | 64/64 |
| D250 | 88.5 | 112.3 | +0.608 | +0.017 | 64/64 |

### mae (54 CV tiles, by family)
| arm | forested | hilly | sparse |
|---|---|---|---|
| C90 | +5.340 | +7.469 | +3.209 |
| B90 | +49.366 | +73.976 | +55.048 |
| D90 | +85.028 | +123.666 | +69.000 |
| C250 | +16.857 | +21.567 | +7.549 |
| B250 | +48.672 | +86.164 | +57.934 |
| D250 | +71.719 | +110.079 | +59.867 |

### rmse (54 CV tiles, by family)
| arm | forested | hilly | sparse |
|---|---|---|---|
| C90 | +7.267 | +10.904 | +6.345 |
| B90 | +65.181 | +97.967 | +74.192 |
| D90 | +106.717 | +155.492 | +89.246 |
| C250 | +22.716 | +30.447 | +14.251 |
| B250 | +63.711 | +111.221 | +77.133 |
| D250 | +90.987 | +138.846 | +78.619 |

### raw_r (54 CV tiles, by family)
| arm | forested | hilly | sparse |
|---|---|---|---|
| C90 | +0.998 | +0.997 | +0.997 |
| B90 | +0.883 | +0.891 | +0.811 |
| D90 | +0.444 | +0.468 | +0.459 |
| C250 | +0.984 | +0.990 | +0.988 |
| B250 | +0.840 | +0.834 | +0.757 |
| D250 | +0.602 | +0.615 | +0.602 |

### resid_r (54 CV tiles, by family)
| arm | forested | hilly | sparse |
|---|---|---|---|
| C90 | +0.000 | +0.000 | +0.000 |
| B90 | -0.026 | -0.030 | +0.004 |
| D90 | -0.046 | -0.039 | -0.016 |
| C250 | +0.000 | +0.000 | +0.000 |
| B250 | +0.085 | +0.031 | +0.043 |
| D250 | +0.034 | +0.008 | +0.016 |

### raw_r > 0 fraction (54 CV tiles)
- C90: 54/54 forested 18/18, hilly 18/18, sparse 18/18
- B90: 54/54 forested 18/18, hilly 18/18, sparse 18/18
- D90: 54/54 forested 18/18, hilly 18/18, sparse 18/18
- C250: 54/54 forested 18/18, hilly 18/18, sparse 18/18
- B250: 54/54 forested 18/18, hilly 18/18, sparse 18/18
- D250: 54/54 forested 18/18, hilly 18/18, sparse 18/18

## Paired block-bootstrap differences (54 CV tiles, blocks resampled)
- B90-C90 resid_r: -0.017 [-0.030, -0.005] n=54
- B90-C90 mae: +54.124 [+45.158, +62.109] n=54
- B90-C90 rmse: +70.941 [+59.045, +81.387] n=54
- B90-C90 raw_r: -0.136 [-0.150, -0.121] n=54
- B250-C250 resid_r: +0.053 [+0.038, +0.069] n=54
- B250-C250 mae: +48.932 [+40.133, +56.752] n=54
- B250-C250 rmse: +61.550 [+50.562, +71.301] n=54
- B250-C250 raw_r: -0.177 [-0.199, -0.155] n=54
- D90-C90 resid_r: -0.034 [-0.043, -0.025] n=54
- D90-C90 mae: +87.225 [+72.989, +100.459] n=54
- D90-C90 rmse: +108.980 [+91.881, +124.868] n=54
- D90-C90 raw_r: -0.541 [-0.574, -0.508] n=54
- D250-C250 resid_r: +0.019 [+0.007, +0.032] n=54
- D250-C250 mae: +65.230 [+54.930, +74.630] n=54
- D250-C250 rmse: +80.346 [+68.069, +91.530] n=54
- D250-C250 raw_r: -0.381 [-0.386, -0.376] n=54
- B90-D90 resid_r: +0.016 [+0.009, +0.024] n=54
- B90-D90 mae: -33.101 [-39.367, -26.119] n=54
- B90-D90 rmse: -38.038 [-45.041, -30.346] n=54
- B90-D90 raw_r: +0.405 [+0.375, +0.436] n=54
- B250-D250 resid_r: +0.034 [+0.025, +0.043] n=54
- B250-D250 mae: -16.298 [-20.088, -12.221] n=54
- B250-D250 rmse: -18.796 [-23.198, -14.044] n=54
- B250-D250 raw_r: +0.204 [+0.184, +0.225] n=54

## Independent reference (USGS 3DEP, CONUS tiles)
- 56 tiles have independent 3DEP truth; others are same-source (COP30/Tilezen == coarse origin).
- C90: n3dep=56 mae=7.1m resid_r=+0.000 raw_r>0=56/56
- B90: n3dep=56 mae=68.3m resid_r=+0.025 raw_r>0=56/56
- D90: n3dep=56 mae=98.9m resid_r=+0.019 raw_r>0=56/56
- C250: n3dep=56 mae=17.1m resid_r=+0.000 raw_r>0=56/56
- B250: n3dep=56 mae=74.5m resid_r=+0.078 raw_r>0=56/56
- D250: n3dep=56 mae=88.0m resid_r=+0.041 raw_r>0=56/56
- paired B-C resid_r (residual detail vs 3DEP, CV tiles, block bootstrap):
  - B90-C90: +0.032 [+0.012, +0.051] n=50
  - B250-C250: +0.080 [+0.065, +0.095] n=50

## Conclusion
- Primary test (native same-source truth, 54 CV tiles, block-bootstrap 95% CI on residual detail):
  - B90-C90 resid_r: -0.017 [-0.030, -0.005] n=54
  - B250-C250 resid_r: +0.053 [+0.038, +0.069] n=54
  - B90-D90 resid_r: +0.016 [+0.009, +0.024] n=54
  - B250-D250 resid_r: +0.034 [+0.025, +0.043] n=54

- Ruling on 'image adds detail beyond prior' (needs B>C and B>D, CI excluding 0 on resid_r):
  - 90 m prior: NO - B90-C90 resid CI entirely <= 0.
  - 250 m prior: weak YES - B250 beats both C and D with CI excluding 0, but delta ~+0.03..+0.05 (a few % of remaining fine-scale variance).
  - Absolute error (MAE/RMSE on held-out 20%): every model arm is 3-8x worse than the simple coarse upsampler (B90 mae 68.8 vs C90 5.8 m; B250 74.8 vs C250 16.6 m), and raw_r of B is below C (0.85/0.81 vs 0.998/0.988).
  - Independent 3DEP truth (56 US tiles): B residual detail is small-positive (+0.025/+0.078) but absolute error is again ~5x the coarse upsampler.

-> VERDICT: **the image adds essentially nothing usable; at best a marginal fine-detail gain at the coarser 250m prior that does not survive the absolute-error comparison.** Conclusion = 'image adds nothing (B ~ C in real terms)'; n=54 CV tiles, 56 tiles with independent USGS 3DEP reference.