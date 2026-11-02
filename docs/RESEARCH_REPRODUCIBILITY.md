# PADDOX research reproducibility workflow

This workflow supports the research manuscript while keeping the existing paper title unchanged.
It does not invent results or replace existing run artifacts.

## Rebuild the predictive datasets

Run from the repository root after the FastF1 collection step has completed:

```bash
python data_pipeline/build_race_sequences.py
python data_pipeline/build_fantasy_features.py
```

The pipeline writes:
- `data/processed/predictive/race_split_manifest.json`
- `data/processed/predictive/fantasy_split_manifest.json`

Each manifest records the race IDs, row counts, and SHA-256 checksum for each split and fails if a race appears in more than one split.

## Train fresh research runs

```bash
python models/lstm_race_trainer.py
python models/rf_fantasy_trainer.py
```

Each new run directory records a `run_manifest.json` with its seed, split checksums, and output filenames. The LSTM run also archives the split manifests when available.

The fantasy trainer now:
- Computes NDCG separately for each race and reports the macro-average across races.
- Shifts relevance by the minimum fantasy score within each race because the PADDOX scoring rules permit negative DNF/DSQ points and NDCG requires non-negative relevance.
- Reports the exact grid-position heuristic (25 - qualifying position) separately from train+validation mean and median baselines.
- Computes paired race-cluster bootstrap confidence intervals for model-minus-baseline MAE differences.

## Generate the paper evidence audit

Use the new run IDs printed by the trainers:

```bash
python scripts/research_reproducibility_audit.py \
  --lstm-run run_<new_lstm_id> \
  --fantasy-run run_<new_fantasy_id>
```

The script writes `reports/predictive/reproducibility_audit.json`. It fails closed if split overlap or strict chronological ordering checks fail, records exact split IDs and hashes, and verifies that every race represented in held-out predictions belongs to the test split (also reporting test-race coverage). It recomputes held-out metrics and distinguishes the grid proxy from training-mean and training-median baselines. Undefined Spearman values are serialized as JSON `null` rather than invalid `NaN` values.

## Important manuscript rule

Do not copy new metric values into the paper until the dataset rebuild, both training runs, and audit have completed successfully. The corrected per-race NDCG values and baseline comparisons may differ from the values in the current draft. The existing paper title is intentionally not changed by this code branch.

## Tests

```bash
pytest -q tests/test_research_metrics.py
```

Run the full project test suite as well before merging this branch. This repository change was authored through GitHub file operations; a local Python environment was not available in this editing session, so the tests are not represented as executed or passed.


## GitHub Actions reproduction run

A manual GitHub Actions workflow is available at
`.github/workflows/paddox-research-reproducibility.yml` on the
`research/reproducibility-audit` branch.

1. Open **Actions → PADDOX Research Reproducibility**.
2. Choose **Run workflow** and select `research/reproducibility-audit`.
3. Set `fetch_data=true` only when the raw FastF1 files are not already present
   in the runner cache/workspace. Fetching can take several hours and may be
   interrupted by upstream rate limits.
4. Download the `paddox-research-<run id>` artifact after the workflow finishes.
   It contains processed splits, model-run outputs, fetch logs, and the audit
   JSON when those files were successfully produced.

The workflow is deliberately manual: it does not run automatically on every
commit and it does not modify the protected/release branch. A green workflow
run is required before reporting newly reproduced metrics. If fetching is
interrupted, inspect the fetch log and rerun with `fetch_data=true`; the fetch
script validates existing session files and resumes missing/invalid sessions.

## Reproduction status

The workflow and code changes are committed, but repository edits alone do not
constitute an experiment run. Record the GitHub Actions run URL, run IDs,
dataset hashes, and audit JSON before updating any paper results. Historical
metrics must remain labelled as historical until independently reproduced.
