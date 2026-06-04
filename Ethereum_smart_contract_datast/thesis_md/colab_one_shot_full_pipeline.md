# Colab — One Notebook, Full Pipeline (Slither + All Ablations + SHAP)

**Runtime:** GPU (T4 or A100), **High-RAM**.  
**Wall time:** ~20–40 hours at full scale (`FAST_MODE=False`). Use **Colab Pro** and keep the tab open.  
**Cross-modal:** built into every `--model hybrid` run (`--hybrid-attention-heads 4`).  
**Cross-contract:** separate hybrid run with `--hybrid-enable-cross-contract`.

Upload to `REPO` on Drive before Cell 2:

- `esc_primary_slither.zip`, `smartbugs_secondary_slither.zip`, `codebert_base.zip` (optional)
- Folders: `contract_dataset_ethereum/`, `smartbugs_wild/contracts/`
- This repo (or `git clone`)

---

## Cell 1 — Mount Drive + GPU check

```python
from google.colab import drive
drive.mount("/content/drive")

import torch
assert torch.cuda.is_available(), "Enable Runtime -> GPU"
print(torch.cuda.get_device_name(0))
```

---

## Cell 2 — Setup (fixed: repo discovery + solc-select, no apt solc)

**Common errors this fixes**

- `ModuleNotFoundError: experiment_utils` → repo path wrong or only zips uploaded (no `.py` files)
- `Unable to locate package solc` → skip `apt-get`; use `solc-select` only

**Option A — run the setup script from the repo:**

```python
%run thesis_md/colab_setup_cell.py
```

**Option B — paste the full cell** from `thesis_md/colab_setup_cell.py` (or copy below).

Quick check before imports:

```python
from pathlib import Path
DRIVE = Path("/content/drive/MyDrive/thesis")
for p in [DRIVE / "Ethereum_smart_contract_datast",
          DRIVE / "Ethereum_smart_contract_datast" / "Ethereum_smart_contract_datast"]:
    print(p, "train_experiment.py =", (p / "train_experiment.py").is_file())
```

You need **`train_experiment.py` = True**. If both are False, clone the repo:

```python
!git clone https://github.com/Yiwen91/Ethereum_smart_contract_datast.git /content/drive/MyDrive/thesis/Ethereum_smart_contract_datast
```

---

## Cell 3 — ONE SHOT: all 10 trainings + SHAP + archive

Paste this entire cell (or run the script file):

```python
import os
os.chdir(REPO)

# Edit paths inside script if Cell 2 paths differ
!python colab_one_shot_full_pipeline.py
```

**Or** inline without the script file — copy from `colab_one_shot_full_pipeline.py` and set at top:

```python
DRIVE = Path("/content/drive/MyDrive/thesis")
REPO  = DRIVE / "Ethereum_smart_contract_datast"
ESC_CONTRACTS = DRIVE / "contract_dataset_ethereum"
SB_CONTRACTS  = DRIVE / "smartbugs_wild/contracts"
RESULTS_DRIVE = DRIVE / "experiment_results_slither_one_shot"
FAST_MODE = False   # True = quick wiring test only
```

Then `%run colab_one_shot_full_pipeline.py` or `exec(open("colab_one_shot_full_pipeline.py").read())` after syncing paths in the script.

---

## What runs (in order)

| # | Job | Model | Dataset | Notes |
|---|-----|-------|---------|-------|
| 1 | `esc_slither` | Slither detectors | ESC | Traditional baseline |
| 2 | `esc_codebert` | CodeBERT | ESC | Ablation A, 100k |
| 3 | `esc_gnn` | GCN | ESC | Ablation B, 100k |
| 4 | `esc_hybrid_crossmodal` | Hybrid + attention | ESC | Ablation C, 100k, **cross-modal** |
| 5 | `esc_hybrid_crosscontract` | Hybrid + `--hybrid-enable-cross-contract` | ESC | C+ |
| 6 | `sb_slither` | Slither | SmartBugs | |
| 7 | `sb_codebert` | CodeBERT | SmartBugs | 100k |
| 8 | `sb_gnn` | GCN | SmartBugs | 5k (graph cost) |
| 9 | `sb_hybrid_crossmodal` | Hybrid | SmartBugs | 5k, cross-modal |
| 10 | `sb_hybrid_crosscontract` | Hybrid + cross-contract | SmartBugs | 5k |
| 11 | SHAP | Hybrid text branch | ESC val | Reentrancy + TOD labels |
| 12 | Archive | — | — | Copy all `*_one_shot` runs to Drive |

Run folders:

- `experiments/hybrid_baseline/esc_hybrid_one_shot` ← main thesis hybrid + SHAP
- `experiments/hybrid_baseline/esc_hybrid_crosscontract_one_shot`

---

## Cell 4 — Results table (after pipeline)

```python
import json
from pathlib import Path

for p in sorted((REPO / "experiments").rglob("test_metrics.json")):
    if "one_shot" not in str(p):
        continue
    m = json.loads(p.read_text())
    print(f"{p.parent.name:42} micro_f1={m.get('micro_f1',0):.4f}  subset={m.get('subset_accuracy',0):.4f}")
```

---

## Expected “good” metrics (full scale, Slither labels)

These are targets from prior tuned runs; your Slither-labeled numbers may differ slightly:

| Run | micro-F1 (test) |
|-----|-----------------|
| ESC CodeBERT | ~0.93 |
| ESC GNN | ~0.90 |
| ESC Hybrid (cross-modal) | **~0.97** |
| SmartBugs CodeBERT | ~0.89 |
| SmartBugs Hybrid 5k | ~0.79 |

---

## If the session disconnects

Results are under `RESULTS_DRIVE` and `experiments/**/_one_shot`.  
Re-run the script after editing `_build_jobs()` to skip completed runs, or run single commands from `colab_end_to_end_slither_experiments.md`.

---

## Single-command reference (manual rerun)

**ESC hybrid (cross-modal, tuned):**

```bash
python train_experiment.py --model hybrid --codebert-model-name hf_models/codebert-base \
  --split-dir experiment_splits/esc_primary_slither \
  --contract-root . --contracts-dir /content/drive/MyDrive/thesis/contract_dataset_ethereum \
  --output-dir experiments/hybrid_baseline --run-name esc_hybrid_one_shot \
  --max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 \
  --sample-strategy reservoir --hybrid-epochs 5 --hybrid-train-batch-size 2 \
  --hybrid-eval-batch-size 4 --max-length 192 --hybrid-max-nodes 96 \
  --hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.10 \
  --hybrid-checkpoint-metric weighted_f1 --device cuda --save-model \
  --threshold-candidates 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8
```

**SHAP (after hybrid saved):**

```bash
python run_shap_explain.py --model hybrid \
  --run-dir experiments/hybrid_baseline/esc_hybrid_one_shot \
  --split-dir experiment_splits/esc_primary_slither --split val \
  --contracts-dir /content/drive/MyDrive/thesis/contract_dataset_ethereum \
  --label "Reentrancy" --max-samples 8 --max-evals 100 --device cuda
```
