# Changelog

Notable changes to the benchmark harness, manifests and the frozen acceptance
configuration. One entry per commit; commit hashes are real, dates are UTC.

## 2026-10-17

- `249fd31` **acceptance.yaml metadata edit** — `primary_metric` renamed from
  `rmse_delta_vs_bicubic` to `improvement_vs_prior_baseline`, with the comment
  rewritten to describe the prior-degradation baseline. **Thresholds were not
  changed.** The rename alters the file sha256 (now `b207b42db69f...`), which
  every benchmark report records; runs that recorded an earlier hash remain
  comparable because no threshold value moved.
- `7f27233` report `unusable_rate = (failed + no_metrics + low_confidence)/n`
  beside `failure_rate` in benchmark.md/JSON; no threshold reads it.
- `525241f` frozen Phase 1b benchmark snapshot committed under
  `docs/benchmarks/phase1b/` (clean-tree runs at `249fd31`).
- `249fd31` prior-degradation (`prior_90m`, `prior_250m`) and `const_mean`
  baselines; finetuned arm excluded from primary-arm choice and acceptance
  checks.
- `abd1a7b` `prediction_near_constant` made scale-free (`std/|mean|` of the
  prediction): the old ratio compared prediction spread against a meter-scale
  reference and fired on every 0..1-normalized prediction.
- `e83ca09` benchmark manifest tracked in-repo, harness post-run reporting
  fixes, first harness test file; tag `phase1-done`.
