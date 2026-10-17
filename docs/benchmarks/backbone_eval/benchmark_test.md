# DepthWizard benchmark

- git `57e402054d4cac734df292499629490946e3234d` (dirty: False), rasterio 1.4.4 / GDAL 3.10.3
- manifest `C:\Users\harsh\Downloads\files\DepthWizard\server\data\manifest.csv` sha256 `9cfa2aa5b118`; acceptance `C:\Users\harsh\Downloads\files\DepthWizard\server\config\acceptance.yaml` sha256 `b207b42db69f`
- split `test`: 26 tiles / 5 sites; block=180 guard=8 calib_cap=2000000 seed=42 prior_res=90,250 m
- primary arm (lowest site-weighted RMSE): `da2`

## Overall (site-weighted means)

| arm | cal RMSE (m) | cal MAE (m) | bias (m) | raw r | raw r (opposite sign) | delta vs baseline (m) | improvement vs baseline | skill vs const | valid/low/failed/no_metrics | failure rate | unusable rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| brightness | 220.435 | 151.92 | -62.9487 | -0.131189 | 0.131189 | 217.525 | -80.4149 | -0.380829 | 5/18/3/0 | 0.115385 | 0.807692 |
| da2 | 154.105 | 105.988 | -47.0357 | -0.0548882 | 0.0548882 | 151.699 | -66.5521 | 0.0346674 | 11/14/1/0 | 0.0384615 | 0.576923 |
| da3-base | 200.498 | 143.498 | -64.0903 | 0.0263506 | -0.0263506 | 197.571 | -73.3627 | -0.255942 | 11/12/3/0 | 0.115385 | 0.576923 |
| da3-large | 165.341 | 117.463 | -50.7347 | -0.0848566 | 0.0848566 | 162.935 | -70.6418 | -0.0357161 | 7/18/1/0 | 0.0384615 | 0.730769 |
| da3-small | 173.263 | 119.357 | -42.7932 | -0.0866973 | 0.0866973 | 170.856 | -72.5092 | -0.0853383 | 14/11/1/0 | 0.0384615 | 0.461538 |
| da3mono-large | 162.438 | 117.007 | -41.9514 | -0.131301 | 0.131301 | 159.609 | -65.4818 | -0.0175311 | 8/15/3/0 | 0.115385 | 0.692308 |
| finetuned * | 191.041 | 130.725 | -55.2906 | 0.0374356 | -0.0374356 | 188.129 | -71.5646 | -0.196702 | 5/19/2/0 | 0.0769231 | 0.807692 |
| imele | 203.169 | 147.951 | -47.0579 | 0.0699871 | -0.0699871 | 200.254 | -74.631 | -0.272675 | 10/13/3/0 | 0.115385 | 0.615385 |
| pix2pix | 215.566 | 152.181 | -57.6358 | -0.14021 | 0.14021 | 212.658 | -79.4767 | -0.350329 | 7/16/3/0 | 0.115385 | 0.730769 |
| sobel | 174.796 | 127.633 | -39.6773 | 0.0112406 | -0.0112406 | 172.423 | -74.4102 | -0.0949405 | 8/15/3/0 | 0.115385 | 0.692308 |

`*` contaminated (trained on val/test sites): excluded from the primary-arm choice and from acceptance checks.

## Baselines (site-weighted)

| baseline | res (m) | value (m) | RMSE (m) | MAE (m) | bias (m) | r vs reference | tiles |
|---|---|---|---|---|---|---|---|
| prior_90m | 90 |  | 2.3145 | 1.4855 | 0.00291729 | 0.999705 | 26 |
| prior_250m | 250 |  | 6.84663 | 4.40112 | 0.00323148 | 0.997829 | 26 |
| const_mean |  | 995.978 | 159.639 | 126.022 | -2.23664 |  | 26 |

Reference-agreement arms (stored coarse DEM scored against itself; agreement, not accuracy): nearest RMSE 0 m, bicubic RMSE 6.39869e-14 m.

## Per stratum (primary arm)

| stratum | tiles | sites | cal RMSE (m) | raw r | raw r (opposite sign) | improvement vs baseline | skill vs const | valid/low/failed/no_metrics | failure rate |
|---|---|---|---|---|---|---|---|---|---|
| forested | 4 | 1 | 185.72 | -0.118416 | 0.118416 | -44.8048 | 0.238958 | 2/2/0/0 | 0 |
| hilly | 4 | 1 | 133.985 | -0.0820655 | 0.0820655 | -36.1984 | -0.166095 | 1/2/1/0 | 0.25 |
| sparse | 18 | 3 | 150.273 | -0.0246529 | 0.0246529 | -83.9192 | -0.0263132 | 8/10/0/0 | 0 |

## Per site (primary arm)

| site | stratum | split | tiles | cal RMSE (m) | raw r | raw r (opposite sign) | delta vs baseline (m) | failure rate |
|---|---|---|---|---|---|---|---|---|
| appalachians_smokies | forested | test | 4 | 185.72 | -0.118416 | 0.118416 | 181.898 | 0 |
| mojave | sparse | test | 3 | 329.624 | 0.36686 | -0.36686 | 327.571 | 0 |
| n_arizona | sparse | test | 14 | 100.753 | 0.0217512 | -0.0217512 | 98.5932 | 0 |
| new_mexico | sparse | test | 1 | 20.4429 | -0.46257 | 0.46257 | 20.0565 | 0 |
| western_ghats | hilly | test | 4 | 133.985 | -0.0820655 | 0.0820655 | 130.373 | 0.25 |

## Bootstrap (95% CI over sites)

| arm | metric | mean | ci low | ci high | n sites | iters |
|---|---|---|---|---|---|---|
| brightness | cal_rmse | 220.435 | 121.532 | 340.479 | 4 | 1000 |
| brightness | raw_r | -0.131189 | -0.204441 | -0.0782758 | 5 | 1000 |
| brightness | raw_r_opposite | 0.131189 | 0.0782758 | 0.204441 | 5 | 1000 |
| brightness | delta_vs_baseline | 217.525 | 118.65 | 337.984 | 4 | 1000 |
| da2 | cal_rmse | 154.105 | 75.1325 | 251.368 | 5 | 1000 |
| da2 | raw_r | -0.0548882 | -0.296875 | 0.18002 | 5 | 1000 |
| da2 | raw_r_opposite | 0.0548882 | -0.18002 | 0.296875 | 5 | 1000 |
| da2 | delta_vs_baseline | 151.699 | 73.3995 | 248.692 | 5 | 1000 |
| da3-base | cal_rmse | 200.498 | 118.933 | 306.495 | 4 | 1000 |
| da3-base | raw_r | 0.0263506 | -0.184088 | 0.23679 | 5 | 1000 |
| da3-base | raw_r_opposite | -0.0263506 | -0.23679 | 0.184088 | 5 | 1000 |
| da3-base | delta_vs_baseline | 197.571 | 116.017 | 304 | 4 | 1000 |
| da3-large | cal_rmse | 165.341 | 76.8311 | 268.868 | 5 | 1000 |
| da3-large | raw_r | -0.0848566 | -0.20756 | 0.035417 | 5 | 1000 |
| da3-large | raw_r_opposite | 0.0848566 | -0.035417 | 0.20756 | 5 | 1000 |
| da3-large | delta_vs_baseline | 162.935 | 75.0981 | 266.191 | 5 | 1000 |
| da3-small | cal_rmse | 173.263 | 75.171 | 290.229 | 5 | 1000 |
| da3-small | raw_r | -0.0866973 | -0.34137 | 0.154222 | 5 | 1000 |
| da3-small | raw_r_opposite | 0.0866973 | -0.154222 | 0.34137 | 5 | 1000 |
| da3-small | delta_vs_baseline | 170.856 | 73.4303 | 287.784 | 5 | 1000 |
| da3mono-large | cal_rmse | 162.438 | 122.246 | 208.992 | 4 | 1000 |
| da3mono-large | raw_r | -0.131301 | -0.213163 | -0.0595293 | 5 | 1000 |
| da3mono-large | raw_r_opposite | 0.131301 | 0.0595293 | 0.213163 | 5 | 1000 |
| da3mono-large | delta_vs_baseline | 159.609 | 119.36 | 206.743 | 4 | 1000 |
| imele | cal_rmse | 203.169 | 119.809 | 300.419 | 4 | 1000 |
| imele | raw_r | 0.0699871 | -0.0829586 | 0.22636 | 5 | 1000 |
| imele | raw_r_opposite | -0.0699871 | -0.22636 | 0.0829586 | 5 | 1000 |
| imele | delta_vs_baseline | 200.254 | 116.916 | 297.924 | 4 | 1000 |
| pix2pix | cal_rmse | 215.566 | 126.319 | 332.763 | 4 | 1000 |
| pix2pix | raw_r | -0.14021 | -0.259097 | -0.00959885 | 5 | 1000 |
| pix2pix | raw_r_opposite | 0.14021 | 0.00959885 | 0.259097 | 5 | 1000 |
| pix2pix | delta_vs_baseline | 212.658 | 123.44 | 330.268 | 4 | 1000 |
| sobel | cal_rmse | 174.796 | 76.244 | 273.347 | 5 | 1000 |
| sobel | raw_r | 0.0112406 | -0.062496 | 0.0982365 | 5 | 1000 |
| sobel | raw_r_opposite | -0.0112406 | -0.0982365 | 0.062496 | 5 | 1000 |
| sobel | delta_vs_baseline | 172.423 | 74.5704 | 270.8 | 5 | 1000 |

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

- arms cover fewer sites than the split has where a calibration fit failed outright (bootstrap 'n sites' column: brightness 4/5, da3-base 4/5, da3mono-large 4/5, imele 4/5, pix2pix 4/5); failed tiles stay in the failure counts.
- improvement/skill baselines: prior_Nm = native reference block-averaged to N m then cubic-resampled back to the working grid; const_mean = mean of the calibration-region reference pixels; skill vs const = 1 - RMSE/RMSE_const (site-weighted); the acceptance baseline is `prior_90m` (the finest --prior-res).
- the nearest/bicubic reference-agreement arms are self-referential on this manifest (RMSE ~ 0: the stored coarse DEM scored against itself) - reported for reference agreement only, NOT the improvement baseline.
- finetuned backend: contaminated (trained on val/test sites) (manifest notes flag test-site validation tiles hilly_15, sparse_02, sparse_04, sparse_07) - its numbers are reported but excluded from the primary-arm choice and acceptance checks.
- 10 tiles use the Tilezen terrarium fallback reference product with an unpublished vertical datum (manifest notes): their absolute errors are not comparable with Copernicus GLO-30 tiles.
- raw-arm MAE/RMSE are in relative-depth units (only correlations and the calibrated arms are comparable across arms); delta vs baseline is defined for the calibrated arm only.
- the random-split columns are the within-scene diagnostic (optimistic by construction); headline numbers are spatial-block with guard band.
- failed / low_confidence tiles remain in every count; the failure rate counts tiles that failed outright (degenerate fit or missing metrics) - unusable_rate = (failed + no_metrics + low_confidence)/n is reported beside it, while warning-carrying tiles are listed in the counts column and the warning-count table above.
- tiles that hit the calibration fit's degenerate/fallback path in this run: hilly_14, hilly_15, sparse_01, sparse_06, sparse_08, sparse_09, sparse_13, sparse_14, sparse_17 (see per_tile.csv calibrated_reason); they stay in every count.
