# DepthWizard benchmark

- git `249fd31dd725fedd70f139a37e1beb85b9566b1e` (dirty: False), rasterio 1.4.4 / GDAL 3.10.3
- manifest `C:\Users\harsh\Downloads\files\DepthWizard\server\data\manifest.csv` sha256 `4febc68feb68`; acceptance `C:\Users\harsh\Downloads\files\DepthWizard\server\config\acceptance.yaml` sha256 `b207b42db69f`
- split `all`: 54 tiles / 7 sites; block=180 guard=8 calib_cap=2000000 seed=42 prior_res=90,250 m
- primary arm (lowest site-weighted RMSE): `da2`

## Overall (site-weighted means)

| arm | cal RMSE (m) | cal MAE (m) | bias (m) | raw r | delta vs baseline (m) | improvement vs baseline | skill vs const | valid/low/failed/no_metrics | failure rate |
|---|---|---|---|---|---|---|---|---|---|
| da2 | 164.311 | 116.801 | -40.2142 | -0.0935779 | 161.878 | -68.1421 | 0.0260447 | 17/34/3/0 | 0.0555556 |
| finetuned * | 190.811 | 135.171 | -50.2667 | 0.0819844 | 188.027 | -71.4928 | -0.131035 | 17/34/3/0 | 0.0555556 |
| imele | 204.246 | 152.33 | -49.1875 | 0.0667471 | 201.463 | -75.6599 | -0.21067 | 22/28/4/0 | 0.0740741 |
| pix2pix | 206.909 | 149.806 | -56.2368 | -0.161326 | 204.126 | -76.7151 | -0.226456 | 14/37/3/0 | 0.0555556 |

`*` contaminated (trained on val/test sites): excluded from the primary-arm choice and from acceptance checks.

## Baselines (site-weighted)

| baseline | res (m) | value (m) | RMSE (m) | MAE (m) | bias (m) | r vs reference | tiles |
|---|---|---|---|---|---|---|---|
| prior_90m | 90 |  | 2.37727 | 1.58642 | 0.00511555 | 0.999742 | 54 |
| prior_250m | 250 |  | 6.76505 | 4.48928 | 0.00343407 | 0.998181 | 54 |
| const_mean |  | 1159.29 | 168.705 | 134.296 | -3.53728 |  | 54 |

Reference-agreement arms (stored coarse DEM scored against itself; agreement, not accuracy): nearest RMSE 0 m, bicubic RMSE 4.5705e-14 m.

## Per stratum (primary arm)

| stratum | tiles | sites | cal RMSE (m) | raw r | improvement vs baseline | failure rate |
|---|---|---|---|---|---|---|
| forested | 18 | 2 | 153.095 | -0.119177 | -47.6809 | 0.0555556 |
| hilly | 18 | 2 | 196.584 | -0.171367 | -64.9377 | 0.111111 |
| sparse | 18 | 3 | 150.273 | -0.0246529 | -83.9192 | 0 |

## Per site (primary arm)

| site | stratum | split | tiles | cal RMSE (m) | raw r | delta vs baseline (m) | failure rate |
|---|---|---|---|---|---|---|---|
| appalachians_smokies | forested | test | 4 | 185.72 | -0.118416 | 181.898 | 0 |
| colorado_front_range | hilly | train | 14 | 259.183 | -0.260668 | 256.514 | 0.0714286 |
| mojave | sparse | test | 3 | 329.624 | 0.36686 | 327.571 | 0 |
| n_arizona | sparse | test | 14 | 100.753 | 0.0217512 | 98.5932 | 0 |
| new_mexico | sparse | test | 1 | 20.4429 | -0.46257 | 20.0565 | 0 |
| nh_white_mountains | forested | train | 14 | 120.47 | -0.119937 | 118.139 | 0.0714286 |
| western_ghats | hilly | test | 4 | 133.985 | -0.0820655 | 130.373 | 0.25 |

## Bootstrap (95% CI over sites)

| arm | metric | mean | ci low | ci high | n sites | iters |
|---|---|---|---|---|---|---|
| da2 | cal_rmse | 164.311 | 98.1637 | 237.762 | 7 | 1000 |
| da2 | raw_r | -0.0935779 | -0.260879 | 0.089386 | 7 | 1000 |
| da2 | delta_vs_baseline | 161.878 | 96.0605 | 234.699 | 7 | 1000 |
| imele | cal_rmse | 204.246 | 137.422 | 267.609 | 6 | 1000 |
| imele | raw_r | 0.0667471 | -0.044467 | 0.187228 | 7 | 1000 |
| imele | delta_vs_baseline | 201.463 | 134.709 | 264.78 | 6 | 1000 |
| pix2pix | cal_rmse | 206.909 | 137.728 | 277.747 | 6 | 1000 |
| pix2pix | raw_r | -0.161326 | -0.265888 | -0.053946 | 7 | 1000 |
| pix2pix | delta_vs_baseline | 204.126 | 135.347 | 275.629 | 6 | 1000 |

## Warning counts (primary arm)

| warning | tiles |
|---|---|
| insufficient_variance | 1 |
| near_zero_scale | 2 |
| negative_scale | 26 |
| ransac_sign_disagreement | 11 |

## Acceptance verdict: **FAIL**

| check | status | value | threshold | reason |
|---|---|---|---|---|
| minimum_improvement | fail | -68.1421 | 0.05 |  |
| per_stratum_improvement.forested | fail | -47.6809 | 0 |  |
| per_stratum_improvement.hilly | fail | -64.9377 | 0 |  |
| per_stratum_improvement.sparse | fail | -83.9192 | 0 |  |
| min_raw_correlation | fail | -0.0935779 | 0.2 |  |
| max_failure_rate | pass | 0.0555556 | 0.1 |  |
| min_test_sites | pass | 5 | 3 |  |

## Caveats

- arms cover fewer sites than the split has where a calibration fit failed outright (bootstrap 'n sites' column: imele 6/7, pix2pix 6/7); failed tiles stay in the failure counts.
- improvement/skill baselines: prior_Nm = native reference block-averaged to N m then cubic-resampled back to the working grid; const_mean = mean of the calibration-region reference pixels; skill vs const = 1 - RMSE/RMSE_const (site-weighted); the acceptance baseline is `prior_90m` (the finest --prior-res).
- the nearest/bicubic reference-agreement arms are self-referential on this manifest (RMSE ~ 0: the stored coarse DEM scored against itself) - reported for reference agreement only, NOT the improvement baseline.
- finetuned backend: contaminated (trained on val/test sites) (manifest notes flag test-site validation tiles hilly_15, sparse_02, sparse_04, sparse_07) - its numbers are reported but excluded from the primary-arm choice and acceptance checks.
- 10 tiles use the Tilezen terrarium fallback reference product with an unpublished vertical datum (manifest notes): their absolute errors are not comparable with Copernicus GLO-30 tiles.
- raw-arm MAE/RMSE are in relative-depth units (only correlations and the calibrated arms are comparable across arms); delta vs baseline is defined for the calibrated arm only.
- the random-split columns are the within-scene diagnostic (optimistic by construction); headline numbers are spatial-block with guard band.
- failed / low_confidence tiles remain in every count; the failure rate counts tiles that failed outright (degenerate fit or missing metrics), while warning-carrying tiles are listed in the counts column and the warning-count table above.
- tiles that hit the calibration fit's degenerate/fallback path in this run: forested_01, forested_03, forested_09, hilly_07, hilly_14, sparse_06, sparse_08, sparse_17 (see per_tile.csv calibrated_reason); they stay in every count.
