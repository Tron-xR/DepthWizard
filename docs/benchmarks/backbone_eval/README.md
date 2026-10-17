# Backbone evaluation: DA3 + control arms (frozen)

Report artifacts of the Phase 2 backbone evaluation, committed for the record.
Live copies are regenerated into `benchmark_runs/` (gitignored) by the harness;
the code changes and design are in `docs/BACKBONE_EVAL_DESIGN.md`.

- run commit: `57e402054d4cac734df292499629490946e3234d` (dirty: False)
- date: 2026-10-17
- manifest sha256 prefix `9cfa2aa5b118`; acceptance sha256 prefix `b207b42db69f`
  (frozen Phase 1b values)
- prediction runs: one clean-worktree run per new arm
  (`benchmark_runs/backbone_eval/57e4020_<arm>_*`), generated with
  `tools/benchmark_tiles.py run --protocol historical --backend <arm>
  --tiles <26 test ids> --no-comparisons`
- scoring command: `python -m app.pipeline.evaluation benchmark --split test`
  with one `--predictions <arm>=<run dir>` override per arm (all 10 arms in a
  single run, so every comparison shares one tile population per tile)

## Results (site-weighted, frozen TEST split: 26 tiles / 5 sites)

| arm | cal RMSE (m) | raw r | raw r (opposite) | skill vs const | failed/26 | unusable | load (s) | peak VRAM (MB) |
|---|---|---|---|---|---|---|---|---|
| da2 (frozen default) | 154.105 | -0.0549 | +0.0549 | +0.035 | 1 | 0.58 | - | - |
| da3mono-large | 162.438 | -0.1313 | +0.1313 | -0.018 | 3 | 0.69 | 6.8 | 2299.7 |
| da3-large (NC) | 165.341 | -0.0849 | +0.0849 | -0.036 | 1 | 0.73 | 9.1 | 2852.5 |
| da3-small | 173.263 | -0.0867 | +0.0867 | -0.085 | 1 | 0.46 | 5.6 | 355.6 |
| sobel (control) | 174.796 | +0.0112 | -0.0112 | -0.095 | 3 | 0.69 | - | - |
| finetuned * | 191.041 | +0.0374 | -0.0374 | -0.197 | 2 | 0.81 | - | - |
| da3-base | 200.498 | +0.0264 | -0.0264 | -0.256 | 3 | 0.58 | 5.4 | 1045.6 |
| imele | 203.169 | +0.0700 | -0.0700 | -0.273 | 3 | 0.62 | - | - |
| pix2pix | 215.566 | -0.1402 | +0.1402 | -0.350 | 3 | 0.73 | - | - |
| brightness (control) | 220.435 | -0.1312 | +0.1312 | -0.381 | 3 | 0.81 | - | - |

Load time is the warm-cache one-time model load (`run.json:model_load_seconds`;
first-ever run also downloads, ~70 s for small); each run's 26 tiles took
58-69 s end to end. `*` contaminated (trained on val/test sites). "NC" = the
weights are CC BY-NC 4.0 (non-commercial). da2/pix2pix/imele/finetuned numbers
are byte-identical to the frozen phase 1b snapshot, which is the sanity gate
this run had to pass.

## Findings

- No DA3 arm beats the frozen default `da2`: best is `da3mono-large` at
  162.438 m vs 154.105 m (and da2 wins even though it is the only arm with
  usable calibrated skill vs const, +0.035).
- Raw correlations are noise-level for every arm (all |r| < 0.15, 95% CI over
  sites spans zero everywhere except `da3mono-large` / `brightness` where only
  the OPPOSITE-sign view is distinguishable from zero). The acceptance verdict
  stays **FAIL**, exactly like frozen phase 1b.
- Controls bound the story: sobel (gradient magnitude, no depth model)
  lands 174.8 m, between the DA3 arms, and brightness 220.4 m lands below
  pix2pix. On this split, trivial image statistics explain at least as much
  score as any depth backbone - the split is too noisy for backbone ranking,
  not evidence for any arm.
- Sign convention: DA3 z-depth ran with `invert_depth=False` per the locked
  family rule (see design doc); `raw_r_opposite` is reported for every arm so a
  family-level inversion mistake would be visible. The observed signs are mixed
  across arms, consistent with noise; no anomaly attributable to the DA3
  adapter.
- All DA3 runs record hub revision + weight sha256 in `model_info.json`
  (e.g. da3-small rev `e08cab65`, sha256 prefix `364492e38a3a`), so the exact
  bytes scored are pinned.
- `da3-giant` was hardware-skipped (1.15 B params ~ 4.6 GB fp32 > 4 GB VRAM on
  this box); `da3-large` peaked at 2.85 GB and fits.

Files:

- `benchmark.md` in the canonical run dir is copied here as `benchmark_test.md`
  (full per-arm, per-stratum, per-site and bootstrap tables; acceptance FAIL).
