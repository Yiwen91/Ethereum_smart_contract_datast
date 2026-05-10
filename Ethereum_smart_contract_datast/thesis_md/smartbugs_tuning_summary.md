# SmartBugs Secondary Tuning Summary

This note records the implemented SmartBugs tuning workflow for the three ablation models and explains how to report the secondary-dataset section in the thesis.

## What was implemented

### 1. Secondary split regeneration

The missing SmartBugs split directory was regenerated from:

- `standardized_smartbugs/standardized_dataset.json`

and saved to:

- `experiment_splits/smartbugs_secondary`

Observed split summary:

- Total contracts: `47,390`
- Total functions: `1,102,546`
- Train contracts/functions: `33,175 / 771,448`
- Validation contracts/functions: `7,114 / 166,365`
- Test contracts/functions: `7,101 / 164,733`

This makes the SmartBugs secondary split reproducible locally rather than relying only on prior Colab sessions.

### 2. SmartBugs preprocessing helper

A dedicated helper was added:

- `prepare_smartbugs_secondary_splits.bat`

and:

- `run_dataset2_smartbugs.bat`

now calls that helper directly so the SmartBugs secondary workflow standardizes the data, writes the vulnerability report, and regenerates the experiment splits in one step.

### 3. Tuning recipes for all three models

The repository now includes tuned SmartBugs recipes in:

- `README.md`
- `thesis_md/smartbugs_tuning_playbook.md`

These recipes follow the best-per-model secondary-dataset strategy:

- `CodeBERT` at larger SmartBugs scale
- `GNN-only` at reduced structural scale
- `Hybrid` at reduced fusion scale

## Local verification runs

Because this machine has no CUDA device, the full tuned SmartBugs runs should still be executed in Colab. However, the secondary-dataset pipelines were verified locally with tiny runs to confirm that the regenerated split and command wiring are valid.

Verified local runs:

- `experiments/codebert_baseline/smartbugs_codebert_tune_tiny_local`
- `experiments/gnn_baseline/smartbugs_gnn_tune_tiny_local`
- `experiments/hybrid_baseline/smartbugs_hybrid_tune_tiny_local`

These local runs are not thesis results. They only confirm that:

- `train_experiment.py` loads `experiment_splits/smartbugs_secondary`
- `CodeBERT`, `GNN`, and `Hybrid` all execute on the regenerated SmartBugs split
- thresholds and metric outputs are written correctly for all three model families

## Final recommended tuned runs

### CodeBERT

Use:

- `experiments/codebert_baseline/smartbugs_codebert_tuned_100k`

Target role in thesis:

- main SmartBugs secondary baseline

### GNN

Use:

- `experiments/gnn_baseline/smartbugs_gnn_tuned_5k`

Target role in thesis:

- reduced-scale structural ablation

### Hybrid

Use:

- `experiments/hybrid_baseline/smartbugs_hybrid_tuned_5k`

Target role in thesis:

- reduced-scale fusion ablation

## Thesis comparison note

Use the following comparison logic in the thesis:

```text
On the SmartBugs Wild secondary dataset, the tuning strategy remained intentionally asymmetric across models because the semantic and structural branches scale very differently in the available Colab environment. CodeBERT was tuned at larger scale and retained its role as the main secondary baseline, while the GNN-only and Hybrid models were tuned at reduced scale due to the dominant AST/CFG graph-extraction cost. This keeps the comparison methodologically honest while still demonstrating that all three ablations were optimized and evaluated on the secondary dataset.
```

## Reporting guidance

When the final Colab runs complete, save for each model:

- `run_config.json`
- `thresholds.json`
- `val_metrics.json`
- `test_metrics.json`
- `summary.txt`
- one thesis-style `*_result_explanation.md`

Then compare each tuned SmartBugs result against its existing baseline note:

- `experiments/codebert_baseline/smartbugs_codebert_100k_result_explanation.md`
- `experiments/gnn_baseline/smartbugs_gnn_5k_result_explanation.md`
- `experiments/hybrid_baseline/smartbugs_hybrid_5k_result_explanation.md`

## Bottom line

The SmartBugs tuning workflow is now implemented in the repo as:

- reproducible secondary splits
- a dedicated SmartBugs split-generation helper
- tuned command recipes for all three models
- local verification that all three secondary-dataset pipelines still run end to end

The remaining expensive step is the actual full tuned execution in Colab, which is now a run-and-collect task rather than a repo-setup task.
