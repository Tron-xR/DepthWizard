# DepthWizard benchmark - ndsm mode

- git `e980dcaf52359058992f99cbe854fd36d1bd181b` (dirty: True)
- height manifest `..\experiments\rs_eval\runs\height_manifest_test.csv` sha256 `0bc15e979101`
- 30 tiles / 8 sites; block=180 guard=8 calib_cap=2000000 seed=42
- headline: predicted nDSM in metres vs the reference on one shared valid mask; skill measured against the predict-zero baseline
- primary arm (lowest site-weighted RMSE): `altimap_v1`

## Overall (site-weighted means)

| arm | headline | RMSE (m) | MAE (m) | bias (m) | r | raw r | raw r (opposite sign) | scale | skill vs predict_zero | delta vs predict_zero (m) | valid/low/failed/no_metrics | failure rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| altimap_v1 | raw | 7.78782 | 5.36207 | -4.97747 | 0.720767 | 0.720767 | -0.720767 | 1.60328 | 0.304206 | -3.40489 | 30/0/0/0 | 0 |
| brightness | calibrated | 8.87784 | 6.72247 | -0.860069 | 0.248085 | -0.0999024 | 0.0999024 | -1.08871 | 0.20682 | -2.94716 | 4/20/6/0 | 0.2 |
| da2 | calibrated | 9.05536 | 6.7644 | -1.03506 | 0.344884 | -0.164932 | 0.164932 | -4.678 | 0.190959 | -3.40297 | 1/17/12/0 | 0.4 |
| da3 | calibrated | 8.93901 | 6.75326 | -1.33005 | 0.164927 | -0.0161196 | 0.0161196 | -1.33646 | 0.201355 | -2.72524 | 3/16/11/0 | 0.366667 |
| rs3dada | raw | 8.67325 | 6.18368 | -5.21199 | 0.580347 | 0.580347 | -0.580347 | 1.77179 | 0.225098 | -2.51946 | 30/0/0/0 | 0 |
| sobel | calibrated | 9.15719 | 7.20823 | -1.2649 | 0.122012 | 0.0758299 | -0.0758299 | 7.62107 | 0.181862 | -2.90995 | 5/13/12/0 | 0.4 |

`headline` = which arm the RMSE column comes from: `raw` = direct metres output (metric checkpoints), `calibrated` = spatial-block affine fit (relative-depth backends/control arms, whose pixels have no metre scale otherwise).

## Baselines (site-weighted)

| baseline | value (m) | RMSE (m) | MAE (m) | bias (m) | r vs reference | tiles |
|---|---|---|---|---|---|---|
| const_mean | 8.1546 | 7.80664 | 6.73275 | 0.319668 |  | 30 |
| predict_zero |  | 11.1927 | 7.84399 | -7.83494 |  | 30 |

## Per stratum (primary arm)

| stratum | tiles | sites | RMSE (m) | bias (m) | raw r | skill vs predict_zero | failure rate |
|---|---|---|---|---|---|---|---|
| forest | 8 | 2 | 11.981 | -8.23509 | 0.676629 | 0.19447 | 0 |
| hilly | 6 | 2 | 7.53043 | -5.1194 | 0.508635 | 0.272575 | 0 |
| sparse | 8 | 2 | 4.63276 | -2.45401 | 0.830951 | 0.336222 | 0 |
| urban | 8 | 2 | 7.00713 | -4.10138 | 0.866851 | 0.442369 | 0 |

## Bootstrap (95% CI over sites)

| arm | metric | mean | ci low | ci high | n sites | iters |
|---|---|---|---|---|---|---|
| altimap_v1 | metric_rmse | 7.78782 | 4.84803 | 10.8595 | 8 | 1000 |
| altimap_v1 | metric_r | 0.720767 | 0.609478 | 0.8133 | 8 | 1000 |
| altimap_v1 | raw_r | 0.720767 | 0.609478 | 0.8133 | 8 | 1000 |
| altimap_v1 | skill_vs_predict_zero | 0.292032 | 0.197588 | 0.371567 | 8 | 1000 |
| brightness | metric_rmse | 8.87784 | 6.13424 | 11.2658 | 8 | 1000 |
| brightness | metric_r | 0.248085 | 0.155511 | 0.331654 | 8 | 1000 |
| brightness | raw_r | -0.0999024 | -0.249777 | 0.0534024 | 8 | 1000 |
| brightness | skill_vs_predict_zero | -0.027259 | -0.535739 | 0.269134 | 8 | 1000 |
| da2 | metric_rmse | 9.05536 | 7.57902 | 10.456 | 8 | 1000 |
| da2 | metric_r | 0.344884 | 0.286492 | 0.405537 | 8 | 1000 |
| da2 | raw_r | -0.164932 | -0.317732 | -0.00167909 | 8 | 1000 |
| da2 | skill_vs_predict_zero | 0.233386 | 0.155355 | 0.318206 | 8 | 1000 |
| da3 | metric_rmse | 8.93901 | 6.48067 | 11.1753 | 8 | 1000 |
| da3 | metric_r | 0.164927 | 0.102244 | 0.225029 | 8 | 1000 |
| da3 | raw_r | -0.0161196 | -0.0957963 | 0.0657133 | 8 | 1000 |
| da3 | skill_vs_predict_zero | -0.0889747 | -0.720023 | 0.266337 | 8 | 1000 |
| rs3dada | metric_rmse | 8.67325 | 5.5973 | 11.9822 | 8 | 1000 |
| rs3dada | metric_r | 0.580347 | 0.479785 | 0.677971 | 8 | 1000 |
| rs3dada | raw_r | 0.580347 | 0.479785 | 0.677971 | 8 | 1000 |
| rs3dada | skill_vs_predict_zero | -0.557471 | -2.16704 | 0.292816 | 8 | 1000 |
| sobel | metric_rmse | 9.15719 | 6.76817 | 11.1064 | 8 | 1000 |
| sobel | metric_r | 0.122012 | 0.075004 | 0.172646 | 8 | 1000 |
| sobel | raw_r | 0.0758299 | 0.0200528 | 0.125779 | 8 | 1000 |
| sobel | skill_vs_predict_zero | 0.0620865 | -0.282376 | 0.264033 | 8 | 1000 |

## Caveats

- rs3dada licence: weights MIT; trained on SynRS3D (dataset licence disputed: HF card CC BY-NC 4.0 vs Zenodo CC BY 4.0) - see docs/RS_HEIGHT_EVAL_DESIGN.md.
- altimap_v1 licence: weights MIT; trained on GAMUS (CC BY 4.0).
- relative-depth arms (headline `calibrated`) are fitted to the reference on the calibration blocks only (spatial split, same guard band as the rdsm protocol); metric arms are scored raw, their `scale` column is a drift diagnostic (~1 expected), never a correction.
- this mode is diagnostic. Acceptance stays the frozen Phase 1b rdsm protocol; nothing here tunes on the test split.
- failed / low_confidence tiles remain in every count; failure_rate = (failed + no_metrics) / tiles.
