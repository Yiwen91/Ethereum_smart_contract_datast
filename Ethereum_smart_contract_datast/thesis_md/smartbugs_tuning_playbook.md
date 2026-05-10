# SmartBugs Tuning Playbook

This note turns the SmartBugs Wild secondary-dataset tuning plan into a reproducible workflow that is realistic for Colab.

## Objective

Tune all three ablations on the secondary dataset with an honest, compute-aware strategy:

- `CodeBERT` at the largest practical SmartBugs scale
- `GNN-only` at reduced structural scale
- `Hybrid` at reduced fusion scale

This is intentionally different from the primary ESC workflow. On SmartBugs Wild, AST/CFG extraction dominates runtime for structural models, so equal-scale tuning across all three models is not methodologically realistic in the available environment.

## Fixed dataset inputs

Secondary-dataset tuning should use:

- raw contracts: `smartbugs_wild/contracts`
- standardized output: `standardized_smartbugs/standardized_dataset.json`
- split directory: `experiment_splits/smartbugs_secondary`

To regenerate the full SmartBugs preprocessing and split pipeline on Windows:

```cmd
prepare_smartbugs_secondary_splits.bat
```

Equivalent direct command:

```cmd
py prepare_experiment_splits.py --from-json standardized_smartbugs\standardized_dataset.json --output-dir experiment_splits\smartbugs_secondary
```

## Baselines already documented

Current SmartBugs baselines already stored in the repo:

- `experiments/codebert_baseline/smartbugs_codebert_100k_result_explanation.md`
- `experiments/gnn_baseline/smartbugs_gnn_5k_result_explanation.md`
- `experiments/hybrid_baseline/smartbugs_hybrid_5k_result_explanation.md`

Those files remain the reference point for before/after comparison.

## Recommended tuning order

1. Tune `CodeBERT` first.
2. Tune `GNN` second with a very small search space.
3. Tune `Hybrid` last using the reduced structural budget already validated by the GNN run.

## Recommended SmartBugs tuning commands

### CodeBERT

Primary tuned candidate:

```cmd
py train_experiment.py --model codebert --codebert-model-name microsoft/codebert-base --split-dir experiment_splits\smartbugs_secondary --output-dir experiments\codebert_baseline --run-name smartbugs_codebert_tuned_100k --max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 --sample-strategy reservoir --epochs 4 --train-batch-size 8 --eval-batch-size 8 --max-length 192 --learning-rate 1.5e-5 --weight-decay 0.01 --max-pos-weight 8 --grad-clip-norm 1.0 --default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.10 --threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 --save-model
```

Fallback candidate if memory or runtime becomes tight:

```cmd
py train_experiment.py --model codebert --codebert-model-name microsoft/codebert-base --split-dir experiment_splits\smartbugs_secondary --output-dir experiments\codebert_baseline --run-name smartbugs_codebert_tuned_100k_fallback --max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 --sample-strategy reservoir --epochs 3 --train-batch-size 8 --eval-batch-size 8 --max-length 128 --learning-rate 2e-5 --weight-decay 0.01 --max-pos-weight 8 --grad-clip-norm 1.0 --default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.10 --threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 --save-model
```

Tuning focus:

- `max_length`: `128` vs `192`
- `epochs`: `3` vs `4`
- `learning_rate`: `2e-5` vs `1.5e-5`
- threshold grid: broaden slightly below `0.5`

### GNN

Primary tuned candidate:

```cmd
py train_experiment.py --model gnn --split-dir experiment_splits\smartbugs_secondary --output-dir experiments\gnn_baseline --run-name smartbugs_gnn_tuned_5k --max-train-samples 5000 --max-val-samples 500 --max-test-samples 500 --sample-strategy reservoir --gnn-epochs 3 --gnn-max-nodes 48 --gnn-feature-dim 256 --gnn-hidden-dim 160 --gnn-num-layers 2 --gnn-dropout 0.10 --gnn-train-batch-size 4 --gnn-eval-batch-size 8 --gnn-learning-rate 7e-4 --gnn-weight-decay 1e-4 --gnn-max-pos-weight 8 --gnn-grad-clip-norm 1.0 --default-threshold 0.5 --threshold-min-support 3 --threshold-min-precision 0.10 --threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6
```

Fallback candidate if graph startup becomes too slow:

```cmd
py train_experiment.py --model gnn --split-dir experiment_splits\smartbugs_secondary --output-dir experiments\gnn_baseline --run-name smartbugs_gnn_tuned_5k_fallback --max-train-samples 5000 --max-val-samples 500 --max-test-samples 500 --sample-strategy reservoir --gnn-epochs 2 --gnn-max-nodes 32 --gnn-feature-dim 256 --gnn-hidden-dim 128 --gnn-num-layers 2 --gnn-dropout 0.20 --gnn-train-batch-size 4 --gnn-eval-batch-size 8 --gnn-learning-rate 1e-3 --gnn-weight-decay 1e-4 --gnn-max-pos-weight 8 --gnn-grad-clip-norm 1.0 --default-threshold 0.5 --threshold-min-support 3 --threshold-min-precision 0.10 --threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6
```

Tuning focus:

- `gnn-max-nodes`: `32` vs `48`
- `gnn-hidden-dim`: `128` vs `160`
- `gnn-dropout`: `0.20` vs `0.10`
- `gnn-learning-rate`: `1e-3` vs `7e-4`

### Hybrid

Primary tuned candidate:

```cmd
py train_experiment.py --model hybrid --codebert-model-name microsoft/codebert-base --split-dir experiment_splits\smartbugs_secondary --output-dir experiments\hybrid_baseline --run-name smartbugs_hybrid_tuned_5k --max-train-samples 5000 --max-val-samples 500 --max-test-samples 500 --sample-strategy reservoir --hybrid-epochs 3 --hybrid-train-batch-size 1 --hybrid-eval-batch-size 2 --max-length 128 --hybrid-max-nodes 48 --hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 --hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.08 --hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1e-5 --hybrid-head-learning-rate 3e-4 --hybrid-weight-decay 0.01 --hybrid-max-pos-weight 10 --hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric weighted_f1 --default-threshold 0.5 --threshold-min-support 3 --threshold-min-precision 0.0 --threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 --save-model
```

Fallback candidate if the graph branch becomes too expensive:

```cmd
py train_experiment.py --model hybrid --codebert-model-name microsoft/codebert-base --split-dir experiment_splits\smartbugs_secondary --output-dir experiments\hybrid_baseline --run-name smartbugs_hybrid_tuned_5k_fallback --max-train-samples 5000 --max-val-samples 500 --max-test-samples 500 --sample-strategy reservoir --hybrid-epochs 3 --hybrid-train-batch-size 1 --hybrid-eval-batch-size 2 --max-length 96 --hybrid-max-nodes 32 --hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 --hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.10 --hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1.5e-5 --hybrid-head-learning-rate 4e-4 --hybrid-weight-decay 0.01 --hybrid-max-pos-weight 10 --hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric weighted_f1 --default-threshold 0.5 --threshold-min-support 3 --threshold-min-precision 0.0 --threshold-candidates 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 --save-model
```

Tuning focus:

- `max_length`: `96` vs `128`
- `hybrid-max-nodes`: `32` vs `48`
- `hybrid-graph-residual-scale`: `0.10` vs `0.08`
- `hybrid-head-learning-rate`: `4e-4` vs `3e-4`
- `hybrid-transformer-learning-rate`: `1.5e-5` vs `1e-5`

## Lightweight local verification

This repository can verify the SmartBugs split and command wiring locally, but the full tuned runs should still be treated as Colab jobs because:

- this machine has no CUDA device
- `GNN` and `Hybrid` incur heavy AST/CFG extraction overhead on SmartBugs Wild
- the tuned `CodeBERT` recipe is designed around large-scale secondary-dataset sampling

## Thesis reporting notes

Recommended reporting structure after the tuned runs complete:

- `CodeBERT`: report as the main SmartBugs secondary baseline
- `GNN`: report as the tuned reduced-scale structural ablation
- `Hybrid`: report as the tuned reduced-scale fusion ablation

Suggested comparison paragraph:

```text
On the SmartBugs Wild secondary dataset, the tuning strategy remained intentionally asymmetric across models because the semantic and structural branches scale very differently in the available Colab environment. CodeBERT was tuned at larger scale and retained its role as the main secondary baseline, while the GNN-only and Hybrid models were tuned at reduced scale due to the dominant AST/CFG graph-extraction cost. This keeps the comparison methodologically honest while still demonstrating that all three ablations were optimized and evaluated on the secondary dataset.
```

## Final output checklist

After each tuned run, save and retain:

- `run_config.json`
- `thresholds.json`
- `val_metrics.json`
- `test_metrics.json`
- `summary.txt`
- one thesis-style `*_result_explanation.md`

If the best SmartBugs `Hybrid` run is later used for qualitative case-study inference, train it with `--save-model` so the saved `model/` directory can be loaded by `run_case_study_inference.py`.
