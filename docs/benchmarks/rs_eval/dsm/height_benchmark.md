# DepthWizard benchmark - dsm mode

- git `e980dcaf52359058992f99cbe854fd36d1bd181b` (dirty: True)
- height manifest `..\experiments\rs_eval\runs\height_manifest_test.csv` sha256 `0bc15e979101`
- 30 tiles / 8 sites; block=180 guard=8 calib_cap=2000000 seed=42
- headline: predicted nDSM in metres vs the reference on one shared valid mask; skill measured against the predict-zero baseline
- primary arm (lowest site-weighted RMSE): `altimap_v1`

## Overall (site-weighted means)

| arm | headline | RMSE (m) | MAE (m) | bias (m) | r | raw r | raw r (opposite sign) | scale | skill vs predict_zero | delta vs predict_zero (m) | valid/low/failed/no_metrics | failure rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| altimap_v1 | raw | 6.53355 | 4.69789 | 2.02897 | 0.960721 | 0.960721 | -0.960721 | 0.987366 | 0.994376 | -1155.18 | 30/0/0/0 | 0 |
| brightness | calibrated | 8.0597 | 6.18798 | -0.31124 | 0.930306 | 0.930306 | -0.930306 | 0.97821 | 0.993062 | -1152.93 | 3/21/0/6 | 0.2 |
| da2 | calibrated | 8.46844 | 6.15594 | -0.166368 | 0.920297 | 0.920297 | -0.920297 | 0.957678 | 0.99271 | -1137.17 | 3/15/0/12 | 0.4 |
| da3 | calibrated | 7.99232 | 5.98861 | -0.554745 | 0.929007 | 0.929007 | -0.929007 | 0.962812 | 0.99312 | -1150.18 | 3/16/0/11 | 0.366667 |
| rs3dada | raw | 7.12096 | 5.20437 | 1.78708 | 0.945495 | 0.945495 | -0.945495 | 0.946048 | 0.99387 | -1154.59 | 30/0/0/0 | 0 |
| sobel | calibrated | 8.26163 | 6.24176 | -1.10174 | 0.908638 | 0.908638 | -0.908638 | 0.944721 | 0.992888 | -1141.47 | 2/16/0/12 | 0.4 |

`headline` = which arm the RMSE column comes from: `raw` = direct metres output (metric checkpoints), `calibrated` = spatial-block affine fit (relative-depth backends/control arms, whose pixels have no metre scale otherwise).

## Baselines (site-weighted)

| baseline | value (m) | RMSE (m) | MAE (m) | bias (m) | r vs reference | tiles |
|---|---|---|---|---|---|---|
| const_mean | 1162.98 | 36.3139 | 30.8784 | 1.90801 |  | 30 |
| oracle |  | 9.31536 | 7.48274 | 7.01807 | 0.977751 | 30 |
| predict_zero |  | 1161.71 | 1161.07 | -1161.07 |  | 30 |
| prior_alone |  | 7.37777 | 5.59621 | -0.827295 | 0.920141 | 30 |

## Per stratum (primary arm)

| stratum | tiles | sites | RMSE (m) | bias (m) | raw r | skill vs predict_zero | failure rate |
|---|---|---|---|---|---|---|---|
| forest | 8 | 2 | 7.47428 | 1.57991 | 0.973201 | 0.991681 | 0 |
| hilly | 6 | 2 | 6.87614 | 2.04948 | 0.991796 | 0.994932 | 0 |
| sparse | 8 | 2 | 4.96423 | 1.87754 | 0.988113 | 0.997457 | 0 |
| urban | 8 | 2 | 6.81956 | 2.60896 | 0.889777 | 0.984484 | 0 |

## Bootstrap (95% CI over sites)

| arm | metric | mean | ci low | ci high | n sites | iters |
|---|---|---|---|---|---|---|
| altimap_v1 | metric_rmse | 6.53355 | 4.42695 | 8.58149 | 8 | 1000 |
| altimap_v1 | metric_r | 0.960721 | 0.922006 | 0.987674 | 8 | 1000 |
| altimap_v1 | raw_r | 0.960721 | 0.922006 | 0.987674 | 8 | 1000 |
| altimap_v1 | skill_vs_predict_zero | 0.991983 | 0.988098 | 0.995575 | 8 | 1000 |
| brightness | metric_rmse | 8.0597 | 5.60443 | 10.3474 | 8 | 1000 |
| brightness | metric_r | 0.930306 | 0.869831 | 0.976719 | 8 | 1000 |
| brightness | raw_r | 0.930306 | 0.869831 | 0.976719 | 8 | 1000 |
| brightness | skill_vs_predict_zero | 0.989865 | 0.984999 | 0.994572 | 8 | 1000 |
| da2 | metric_rmse | 8.46844 | 6.53925 | 10.3148 | 8 | 1000 |
| da2 | metric_r | 0.920297 | 0.826245 | 0.980584 | 8 | 1000 |
| da2 | raw_r | 0.920297 | 0.826245 | 0.980584 | 8 | 1000 |
| da2 | skill_vs_predict_zero | 0.989525 | 0.984193 | 0.994169 | 8 | 1000 |
| da3 | metric_rmse | 7.99232 | 5.63072 | 10.0799 | 8 | 1000 |
| da3 | metric_r | 0.929007 | 0.854108 | 0.980226 | 8 | 1000 |
| da3 | raw_r | 0.929007 | 0.854108 | 0.980226 | 8 | 1000 |
| da3 | skill_vs_predict_zero | 0.989967 | 0.984798 | 0.994662 | 8 | 1000 |
| rs3dada | metric_rmse | 7.12096 | 4.83837 | 9.21866 | 8 | 1000 |
| rs3dada | metric_r | 0.945495 | 0.887269 | 0.985785 | 8 | 1000 |
| rs3dada | raw_r | 0.945495 | 0.887269 | 0.985785 | 8 | 1000 |
| rs3dada | skill_vs_predict_zero | 0.990982 | 0.986126 | 0.995335 | 8 | 1000 |
| sobel | metric_rmse | 8.26163 | 5.8815 | 10.6302 | 8 | 1000 |
| sobel | metric_r | 0.908638 | 0.791987 | 0.979792 | 8 | 1000 |
| sobel | raw_r | 0.908638 | 0.791987 | 0.979792 | 8 | 1000 |
| sobel | skill_vs_predict_zero | 0.989377 | 0.983544 | 0.994455 | 8 | 1000 |

## Caveats

- rs3dada licence: weights MIT; trained on SynRS3D (dataset licence disputed: HF card CC BY-NC 4.0 vs Zenodo CC BY 4.0) - see docs/RS_HEIGHT_EVAL_DESIGN.md.
- altimap_v1 licence: weights MIT; trained on GAMUS (CC BY 4.0).
- dsm mode composes arms as prior + predicted nDSM and never subtracts from the prior. Double-counting check: `oracle` = prior + true nDSM (the ceiling if the nDSM were perfect); if an arm's RMSE approaches `prior_alone`, its extra height mostly re-explains terrain the prior already carries.
- relative-depth arms (headline `calibrated`) are fitted to the reference on the calibration blocks only (spatial split, same guard band as the rdsm protocol); metric arms are scored raw, their `scale` column is a drift diagnostic (~1 expected), never a correction.
- this mode is diagnostic. Acceptance stays the frozen Phase 1b rdsm protocol; nothing here tunes on the test split.
- failed / low_confidence tiles remain in every count; failure_rate = (failed + no_metrics) / tiles.
