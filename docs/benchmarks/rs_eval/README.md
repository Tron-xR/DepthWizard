# RS height eval: final run (Phase 3, frozen test split)

Gated diagnostic run of the opt-in remote-sensing height backends against the
Phase 2 HR dataset, per the plan in `docs/RS_HEIGHT_EVAL_DESIGN.md`. Acceptance
stays the frozen Phase 1b rDSM protocol; nothing here was tuned on the test
split and `server/config/acceptance.yaml` is unchanged.

## Setup

- Code: `remote-sensing-eval` rebased onto `backbone-eval` @ `02bfae2`; run
  commit `e980dca`. Dataset commit `088f107` (112 tiles, manifest sha in the
  JSON payloads) - read-only, paths referenced from the main checkout.
- Frozen test split, temporal-flagged excluded like `hr_headroom.md`:
  **30 tiles / 8 sites** (napf_00, napf_01 excluded). Strata: forest 8,
  hilly 6, sparse 8, urban 8.
- Evaluation input: `height_manifest_test.csv` (sha256 `0bc15e97...`), built
  from `server/data/manifest_hr.csv` by `experiments/rs_eval/
  run_hr_predictions.py` (the generator and the run dirs' `run.json` pin
  every prediction's provenance: input sha256, model sha256/hub revision,
  inference fingerprint).
- Arms: `rs3dada` (JTRNEO/RS3DAda) and `altimap_v1` (Dilavesh/altimap-height
  `best.pth`) emit nDSM metres; `da2` (locked default, depth-anything-small)
  and `da3` (DA3-SMALL) emit relative depth; `sobel` / `brightness` are
  trivial controls. Baselines: `predict_zero`, `const_mean` (in ndsm mode),
  plus `oracle` (prior + true nDSM) and `prior_alone` in dsm mode.
- altimap_v2 was NOT run (non-commercial training data); RS3DAda dataset
  licence dispute stays flagged in the design doc.

## Compute

RTX 3050 Laptop 4 GB, `.venv-rs` (python 3.10.20, torch 2.5.1+cu124,
numpy 1.26.4). RS arms ran CUDA bf16 with `DEPTHWIZARD_CUDA_MEM_FRACTION=0.85`;
DA3 package installed `--no-deps` from upstream @ `3d835ec`.

| arm | device | 30 tiles (sum) | peak VRAM |
|---|---|---|---|
| rs3dada | cuda | 9.7 s | 2589 MB |
| altimap_v1 | cuda | 294.6 s (518/259 windows on the 0.33 m/px grid) | 2705 MB |
| da2 | cuda | 3.4 s | 220 MB |
| da3 | cuda | 6.3 s | 356 MB |
| sobel / brightness | cpu | 0.6 s / 0.3 s | - |

## ndsm mode (predicted nDSM vs Swiss nDSM)

Site-weighted, 30 tiles, bootstrap CIs over 8 sites; full tables in
`ndsm/height_benchmark.md`.

| arm | RMSE (m) | bias (m) | raw r | skill vs zero | 95% CI RMSE | valid/low/fail/no_metrics |
|---|---|---|---|---|---|---|
| altimap_v1 | 7.79 | -4.98 | 0.721 | +0.304 | [4.85, 10.86] | 30/0/0/0 |
| rs3dada | 8.67 | -5.21 | 0.580 | +0.225 | [5.60, 11.98] | 30/0/0/0 |
| da3 | 8.94 | -1.33 | -0.016 | +0.201 | [6.48, 11.18] | 3/16/11/0 |
| brightness | 8.88 | -0.86 | -0.100 | +0.207 | [6.13, 11.27] | 4/20/6/0 |
| da2 | 9.06 | -1.04 | -0.165 | +0.191 | [7.58, 10.46] | 1/17/12/0 |
| sobel | 9.16 | -1.26 | +0.076 | +0.182 | [6.77, 11.11] | 5/13/12/0 |
| const_mean | 7.81 | +0.32 | - | - | - | baseline |
| predict_zero | 11.19 | -7.83 | - | - | - | baseline |

Per stratum (metric_rmse, site-weighted; full in the JSON):

| stratum | predict_zero | const_mean | altimap_v1 | rs3dada | da2 | da3 | sobel | brightness |
|---|---|---|---|---|---|---|---|---|
| forest | 14.87 | 10.11 | 11.98 | 12.82 | 11.37 | 12.55 | 11.90 | 12.19 |
| hilly | 7.61 | 4.91 | 7.53 | 8.50 | 8.33 | 7.38 | 7.55 | 7.60 |
| sparse | 6.98 | 5.57 | 4.63 | 5.78 | 6.46 | 5.64 | 7.81 | 6.36 |
| urban | 12.57 | 9.18 | 7.01 | 7.59 | 10.06 | 10.18 | 9.37 | 9.35 |

Read: metric arms (raw metres, no fitting) carry the signal; the relative
arms/controls mostly fail their spatial-block calibration (failure rates
0.2-0.4 - flat/low-contrast tiles), and their raw r is uninformative. The
nDSM signal itself is thin: `const_mean` (~8.15 m everywhere) nearly ties the
best arm overall and beats every arm in hilly and forest.

## dsm mode (prior + predicted nDSM vs Swiss DSM)

Full tables in `dsm/height_benchmark.md`. Baselines first: `prior_alone`
7.38 m, `oracle` (prior + true nDSM) 9.32 m - the oracle is WORSE than the
prior alone, i.e. the crude prior's terrain error exceeds the true canopy
term it omits; arm RMSEs sit in the same 6.5-8.5 m band as that pair, so dsm
composition differences are dominated by how each arm's height field
interacts with prior error, not by real elevation skill.

| arm | RMSE (m) | r | skill vs zero |
|---|---|---|---|
| altimap_v1 | 6.53 | 0.961 | 0.994 |
| rs3dada | 7.12 | 0.945 | 0.994 |
| da3 | 7.99 | 0.929 | 0.993 |
| brightness | 8.06 | 0.930 | 0.993 |
| sobel | 8.26 | 0.909 | 0.993 |
| da2 | 8.47 | 0.920 | 0.993 |

## Answer: urban stratum, ndsm mode

**Yes - `altimap_v1` (7.01 m) and `rs3dada` (7.59 m) beat predict-zero
(12.57 m) and both controls (sobel 9.37 m, brightness 9.35 m) on the urban
stratum; `da2`/`da3` and the controls themselves do not beat the controls'
own bar there.** The caveat is the constant predictor: urban `const_mean`
9.18 m, so only the two metric RS arms clear a trivial constant, and overall
(7.79 vs 7.81 m) altimap_v1 merely ties it.

## Test status

Worktree suite (`.venv-rs`): 207 collected, 205 passed, 3 skipped, 2 failures,
both environmental and on files bit-identical to the frozen base:
`test_shipped_hr_manifest_validates` (the 11 GB `hr/` blob exists only in the
main checkout; passes there, exit 0) and `test_compute_metrics` (float32/
float64 equality assert that flips under the ad-hoc venv's numpy 1.26.4;
passes under the locked env numpy 2.2.6). flake8 (`--max-line-length=100`)
clean on every touched file.

## Reproduce

```
# main checkout: server/data/manifest_hr.csv must exist (read-only use)
.venv-rs\Scripts\python.exe experiments\rs_eval\run_hr_predictions.py \
    --main-manifest ..\DepthWizard\server\data\manifest_hr.csv --only-manifest
# then once per backend: --backend {rs3dada,altimap_v1,da2,da3,sobel,brightness}
cd server
..\.venv-rs\Scripts\python.exe -m app.pipeline.evaluation benchmark \
    --mode {ndsm,dsm} --height-manifest ..\experiments\rs_eval\runs\height_manifest_test.csv \
    --predictions <arm>=..\experiments\rs_eval\runs\<run dir> ... \
    --out ..\docs\benchmarks\rs_eval\{ndsm,dsm}
```
