# DepthWizard benchmark

- git `249fd31dd725fedd70f139a37e1beb85b9566b1e` (dirty: False), rasterio 1.4.4 / GDAL 3.10.3
- manifest `C:\Users\harsh\Downloads\files\DepthWizard\server\data\manifest.csv` sha256 `4febc68feb68`; acceptance `C:\Users\harsh\Downloads\files\DepthWizard\server\config\acceptance.yaml` sha256 `b207b42db69f`
- split `test`: 26 tiles / 5 sites; block=180 guard=8 calib_cap=2000000 seed=42 prior_res=90,250 m
- primary arm (lowest site-weighted RMSE): `da2`

## Overall (site-weighted means)

| arm | cal RMSE (m) | cal MAE (m) | bias (m) | raw r | delta vs baseline (m) | improvement vs baseline | skill vs const | valid/low/failed/no_metrics | failure rate |
|---|---|---|---|---|---|---|---|---|---|
| da2 | 154.105 | 105.988 | -47.0357 | -0.0548882 | 151.699 | -66.5521 | 0.0346674 | 11/14/1/0 | 0.0384615 |
| finetuned * | 191.041 | 130.725 | -55.2906 | 0.0374356 | 188.129 | -71.5646 | -0.196702 | 5/19/2/0 | 0.0769231 |
| imele | 203.169 | 147.951 | -47.0579 | 0.0699871 | 200.254 | -74.631 | -0.272675 | 10/13/3/0 | 0.115385 |
| pix2pix | 215.566 | 152.181 | -57.6358 | -0.14021 | 212.658 | -79.4767 | -0.350329 | 7/16/3/0 | 0.115385 |

`*` contaminated (trained on val/test sites): excluded from the primary-arm choice and from acceptance checks.

## Baselines (site-weighted)

| baseline | res (m) | value (m) | RMSE (m) | MAE (m) | bias (m) | r vs reference | tiles |
|---|---|---|---|---|---|---|---|
| prior_90m | 90 |  | 2.3145 | 1.4855 | 0.00291729 | 0.999705 | 26 |
| prior_250m | 250 |  | 6.84663 | 4.40112 | 0.00323148 | 0.997829 | 26 |
| const_mean |  | 995.978 | 159.639 | 126.022 | -2.23664 |  | 26 |

Reference-agreement arms (stored coarse DEM scored against itself; agreement, not accuracy): nearest RMSE 0 m, bicubic RMSE 6.39869e-14 m.

## Per stratum (primary arm)

| stratum | tiles | sites | cal RMSE (m) | raw r | improvement vs baseline | failure rate |
|---|---|---|---|---|---|---|
| forested | 4 | 1 | 185.72 | -0.118416 | -44.8048 | 0 |
| hilly | 4 | 1 | 133.985 | -0.0820655 | -36.1984 | 0.25 |
| sparse | 18 | 3 | 150.273 | -0.0246529 | -83.9192 | 0 |

## Per site (primary arm)

| site | stratum | split | tiles | cal RMSE (m) | raw r | delta vs baseline (m) | failure rate |
|---|---|---|---|---|---|---|---|
| appalachians_smokies | forested | test | 4 | 185.72 | -0.118416 | 181.898 | 0 |
| mojave | sparse | test | 3 | 329.624 | 0.36686 | 327.571 | 0 |
| n_arizona | sparse | test | 14 | 100.753 | 0.0217512 | 98.5932 | 0 |
| new_mexico | sparse | test | 1 | 20.4429 | -0.46257 | 20.0565 | 0 |
| western_ghats | hilly | test | 4 | 133.985 | -0.0820655 | 130.373 | 0.25 |

## Bootstrap (95% CI over sites)

| arm | metric | mean | ci low | ci high | n sites | iters |
|---|---|---|---|---|---|---|
| da2 | cal_rmse | 154.105 | 75.1325 | 251.368 | 5 | 1000 |
| da2 | raw_r | -0.0548882 | -0.296875 | 0.18002 | 5 | 1000 |
| da2 | delta_vs_baseline | 151.699 | 73.3995 | 248.692 | 5 | 1000 |
| imele | cal_rmse | 203.169 | 119.809 | 300.419 | 4 | 1000 |
| imele | raw_r | 0.0699871 | -0.0829586 | 0.22636 | 5 | 1000 |
| imele | delta_vs_baseline | 200.254 | 116.916 | 297.924 | 4 | 1000 |
| pix2pix | cal_rmse | 215.566 | 126.319 | 332.763 | 4 | 1000 |
| pix2pix | raw_r | -0.14021 | -0.259097 | -0.00959885 | 5 | 1000 |
| pix2pix | delta_vs_baseline | 212.658 | 123.44 | 330.268 | 4 | 1000 |

## Warning counts (primary arm)

| warning | tiles |
|---|---|
| insufficient_variance | 1 |
| negative_scale | 11 |
| ransac_sign_disagreement | 6 |

## Acceptance verdict: **FAIL**

| check | status | value | threshold | reason |
|---|---|---|---|---|
| minimum_improvement | fail | -66.5521 | 0.05 |  |
| per_stratum_improvement.forested | fail | -44.8048 | 0 |  |
| per_stratum_improvement.hilly | fail | -36.1984 | 0 |  |
| per_stratum_improvement.sparse | fail | -83.9192 | 0 |  |
| min_raw_correlation | fail | -0.0548882 | 0.2 |  |
| max_failure_rate | pass | 0.0384615 | 0.1 |  |
| min_test_sites | pass | 5 | 3 |  |

## Caveats

- arms cover fewer sites than the split has where a calibration fit failed outright (bootstrap 'n sites' column: imele 4/5, pix2pix 4/5); failed tiles stay in the failure counts.
- improvement/skill baselines: prior_Nm = native reference block-averaged to N m then cubic-resampled back to the working grid; const_mean = mean of the calibration-region reference pixels; skill vs const = 1 - RMSE/RMSE_const (site-weighted); the acceptance baseline is `prior_90m` (the finest --prior-res).
- the nearest/bicubic reference-agreement arms are self-referential on this manifest (RMSE ~ 0: the stored coarse DEM scored against itself) - reported for reference agreement only, NOT the improvement baseline.
- finetuned backend: contaminated (trained on val/test sites) (manifest notes flag test-site validation tiles hilly_15, sparse_02, sparse_04, sparse_07) - its numbers are reported but excluded from the primary-arm choice and acceptance checks.
- 10 tiles use the Tilezen terrarium fallback reference product with an unpublished vertical datum (manifest notes): their absolute errors are not comparable with Copernicus GLO-30 tiles.
- raw-arm MAE/RMSE are in relative-depth units (only correlations and the calibrated arms are comparable across arms); delta vs baseline is defined for the calibrated arm only.
- the random-split columns are the within-scene diagnostic (optimistic by construction); headline numbers are spatial-block with guard band.
- failed / low_confidence tiles remain in every count; the failure rate counts tiles that failed outright (degenerate fit or missing metrics), while warning-carrying tiles are listed in the counts column and the warning-count table above.
- tiles that hit the calibration fit's degenerate/fallback path in this run: hilly_14, sparse_06, sparse_08, sparse_17 (see per_tile.csv calibrated_reason); they stay in every count.
