# Phase 1b benchmark snapshot (frozen)

Report artifacts of the Phase 1b clean-tree runs, committed for the record.
Live copies are regenerated into `results/` (gitignored) by the harness.

- run commit: `249fd31dd725fedd70f139a37e1beb85b9566b1e` (dirty: False)
- date: 2026-10-17
- manifest sha256 prefix `4febc68feb68`; acceptance sha256 prefix `b207b42db69f`
- command: `uv run python -m app.pipeline.evaluation benchmark --split test --out ..\results`
  (and `--split all --out ..\results\all`), run from `server/`

Files:

- `benchmark_test.md` — `--split test` (26 tiles / 5 sites)
- `benchmark_all.md` — `--split all` (54 tiles / 7 sites)

Note: this snapshot predates the follow-up commits (unusable_rate column,
manifest-driven `contaminated_arms`, changelog) and therefore does not show them.
