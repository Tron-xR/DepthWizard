# HR dataset headroom (Phase 2)

Generated 2026-10-18T23:26:04 from `server/data/manifest_hr.csv`.
All arms scored against `dsm_hr` (the manifest `reference_hr_path`) on
each tile's built valid mask; oracle corrections are fitted on spatial
calibration blocks, metrics on held-out evaluation blocks.
Primary set: 30 test tiles; temporal-flagged excluded: ['napf_00', 'napf_01'].

## Priors vs dsm_hr (RMSE m, overall median of per-tile values)

| arm | median RMSE | median MAE |
| --- | --- | --- |
| coarse_dem | 8.07 | 5.97 |
| coarse_dem_affine | 8.03 | 5.99 |
| coarse_dem_offset_corrected | 8.79 | 6.21 |
| coarse_dem_zeromean | 8.00 | 6.04 |
| const_mean | 35.16 | 30.28 |
| perfect_terrain_floor | 8.59 | 7.47 |
| prior_250m | 13.82 | 10.87 |
| prior_250m_affine | 14.17 | 11.49 |
| prior_250m_zeromean | 14.30 | 11.49 |
| prior_90m | 8.43 | 6.58 |
| prior_90m_affine | 8.47 | 6.63 |
| prior_90m_plus_true_ndsm | 8.09 | 6.30 |
| prior_90m_plus_true_residual | 0.00 | 0.00 |
| prior_90m_zeromean | 8.49 | 6.64 |

### Per stratum (median RMSE m)

| arm | forest | hilly | sparse | urban |
| --- | --- | --- | --- | --- |
| coarse_dem | 8.58 | 3.21 | 5.38 | 8.74 |
| coarse_dem_affine | 8.67 | 3.17 | 5.40 | 9.06 |
| coarse_dem_offset_corrected | 8.93 | 3.20 | 5.70 | 10.77 |
| coarse_dem_zeromean | 8.65 | 3.20 | 5.44 | 8.56 |
| const_mean | 33.72 | 35.88 | 52.92 | 13.96 |
| perfect_terrain_floor | 11.16 | 3.81 | 6.32 | 9.31 |
| prior_250m | 13.15 | 16.43 | 16.03 | 10.62 |
| prior_250m_affine | 12.95 | 15.89 | 15.27 | 11.99 |
| prior_250m_zeromean | 13.38 | 16.46 | 17.43 | 10.52 |
| prior_90m | 9.37 | 5.10 | 7.14 | 9.18 |
| prior_90m_affine | 9.46 | 4.87 | 6.97 | 9.85 |
| prior_90m_plus_true_ndsm | 13.41 | 6.10 | 7.44 | 7.72 |
| prior_90m_plus_true_residual | 0.00 | 0.00 | 0.00 | 0.00 |
| prior_90m_zeromean | 9.43 | 5.12 | 7.12 | 8.86 |

### Variance decomposition (median over primary tiles)

| stratum | std(dsm) | std(dtm) | std(ndsm) | canopy share of var |
| --- | --- | --- | --- | --- |
| forest | 33.9 | 26.5 | 11.1 | 0.13 |
| hilly | 35.9 | 38.0 | 3.8 | -0.06 |
| sparse | 51.2 | 50.7 | 6.3 | 0.00 |
| urban | 13.9 | 10.2 | 9.2 | 0.45 |

## Backends vs dsm_hr (deployed calibration workflow)

Far out of the backends' training domain; a floor expectation for Phase 3, not a tuning target.

| backend | raw r (median) | cal RMSE m (median) | cal r (median) |
| --- | --- | --- | --- |
| pix2pix | 0.00 | 34.1 | 0.25 |
| depth_anything | -0.55 | 23.1 | 0.57 |
| finetuned | 0.49 | 24.7 | 0.53 |
| imele | 0.16 | 36.4 | 0.15 |

## Notes

- `coarse_dem_offset_corrected` adds the per-tile recorded `datum_offset_m` (never removed silently; both arms listed).
- `*_zeromean` / `*_affine` are oracle upper bounds (bias or affine fit estimated on calibration blocks only).
- `perfect_terrain_floor` = dtm + constant canopy mean: the error a perfect terrain model with no canopy awareness leaves.
- `prior_90m_plus_true_ndsm` = prior_90m + built ndsm: where a perfect depth model on top of the 90 m prior lands (its terrain error untouched; oracle, uses reference heights).
- `prior_90m_plus_true_residual` = prior_90m + built residual (= dsm_hr by construction): RMSE near 0 is a plumbing sanity check, not a result.
- Datum offsets in forest sites reflect canopy-blur in the coarse DEM over small clearings, not only the geoid; see docs/DATA_SOURCES.md.
