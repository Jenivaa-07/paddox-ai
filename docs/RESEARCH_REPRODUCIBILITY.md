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

The script writes `reports/predictive/reproducibility_audit.json`. It verifies split overlap and chronological ordering, records exact split IDs and hashes, recomputes held-out metrics, and distinguishes the grid proxy from training-mean and training-median baselines.

## Important manuscript rule

Do not copy new metric values into the paper until the dataset rebuild, both training runs, and audit have completed successfully. The corrected per-race NDCG values and baseline comparisons may differ from the values in the current draft. The existing paper title is intentionally not changed by this code branch.

## Tests

```bash
pytest -q tests/test_research_metrics.py
```

Run the full project test suite as well before merging this branch. This repository change was authored through GitHub file operations; a local Python environment was not available in this editing session, so the tests are not represented as executed or passed.
