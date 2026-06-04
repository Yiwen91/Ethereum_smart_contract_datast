# Google Colab — End-to-End Experiments (Slither Labels)

Copy each **Cell** into a separate Colab code cell, top to bottom.  
**Cross-modal attention** is built into `--model hybrid` (`--hybrid-attention-heads`).  
**Cross-contract** is enabled only with `--hybrid-enable-cross-contract`.

---

## What you must have on Drive (or upload to `/content`)

| Item | Typical path on Drive |
|------|------------------------|
| Repo (this project) | `MyDrive/thesis/Ethereum_smart_contract_datast` |
| ESC `.sol` files | `.../contract_dataset_ethereum/` |
| SmartBugs `.sol` files | `.../smartbugs_wild/contracts/` |
| Slither splits (upload zips) | `esc_primary_slither.zip`, `smartbugs_secondary_slither.zip` (or legacy `esc_primary.zip` = same ESC pack) |
| CodeBERT weights (optional offline) | `codebert_base.zip` → unzip to repo root |

**Unpack splits on Colab** (after uploading zips to `REPO`):

```python
!unzip -q -o esc_primary_slither.zip -d experiment_splits
!unzip -q -o smartbugs_secondary_slither.zip -d experiment_splits
# optional offline transformer (else uses Hugging Face download):
# !unzip -q -o codebert_base.zip -d .
# then: --codebert-model-name hf_models/codebert-base
```

Rebuild zips locally after re-splitting: `python pack_colab_zips.py`

---

## Cell 1 — GPU + Drive

```python
# Runtime: GPU (T4 or better). High-RAM helps for GNN/Hybrid graph cache.
from google.colab import drive
drive.mount("/content/drive")

import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

---

## Cell 2 — Paths (edit these)

```python
import os
from pathlib import Path

# --- EDIT ---
DRIVE = Path("/content/drive/MyDrive/thesis")   # your Drive folder
REPO  = DRIVE / "Ethereum_smart_contract_datast"  # folder containing train_experiment.py

# Contract roots (must contain .sol files referenced in split JSON)
ESC_CONTRACTS = DRIVE / "contract_dataset_ethereum"
SB_CONTRACTS  = DRIVE / "smartbugs_wild" / "contracts"   # or .../smartbugs_wild if layout differs

# Results saved back to Drive
RESULTS_DRIVE = DRIVE / "experiment_results_slither"
RESULTS_DRIVE.mkdir(parents=True, exist_ok=True)

os.chdir(REPO)
print("CWD:", Path.cwd())
print("Repo OK:", (REPO / "train_experiment.py").is_file())
print("ESC splits:", (REPO / "experiment_splits/esc_primary_slither/train.json").is_file())
print("SB splits:", (REPO / "experiment_splits/smartbugs_secondary_slither/train.json").is_file())
print("ESC contracts:", ESC_CONTRACTS.is_dir(), "count~", len(list(ESC_CONTRACTS.rglob("*.sol"))[:5]), "...")
print("SB contracts:", SB_CONTRACTS.is_dir())
```

If the repo is not on Drive yet:

```python
# Optional: clone from GitHub instead of Drive copy
# !git clone https://github.com/Yiwen91/Ethereum_smart_contract_datast.git /content/Ethereum_smart_contract_datast
# REPO = Path("/content/Ethereum_smart_contract_datast")
# os.chdir(REPO)
```

---

## Cell 3 — Install + path check (use fixed setup)

Do **not** use `apt-get install solc` on Colab (often fails). Use:

```python
%run thesis_md/colab_setup_cell.py
```

Or paste all of `thesis_md/colab_setup_cell.py` into one cell. It auto-finds the repo folder, adds `sys.path`, installs `solc-select`, and verifies contract resolution.

---

## Cell 4 — Path resolution smoke test (critical)

If `resolved` is 0, training will produce garbage metrics. Fix `ESC_CONTRACTS` / `SB_CONTRACTS` before long runs.

```python
import json
from experiment_utils import load_named_split

def check_split(split_dir, contracts_dir, n=200):
    split_dir = REPO / split_dir
    s = load_named_split(
        "val",
        split_dir / "val.json",
        max_samples=n,
        seed=42,
        sample_strategy="head",
        project_root=REPO,
        contracts_dir=contracts_dir,
    )
    ok = sum(1 for p in s.contract_paths if p and Path(p).is_file())
    miss = len(s.contract_paths) - ok
    print(split_dir.name, f"resolved={ok}/{len(s.contract_paths)} missing={miss}")
    if ok == 0:
        print("  FIX contracts_dir:", contracts_dir)

check_split("experiment_splits/esc_primary_slither", ESC_CONTRACTS)
check_split("experiment_splits/smartbugs_secondary_slither", SB_CONTRACTS)
```

---

## Cell 5 — Helper to run one experiment

```python
import subprocess
import shlex
from datetime import datetime

def run_train(cmd: str, log_name: str):
    log_path = RESULTS_DRIVE / f"{log_name}.log"
    print("=" * 72)
    print(datetime.now(), log_name)
    print(cmd)
    print("=" * 72)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            shlex.split(cmd),
            cwd=str(REPO),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print("exit", proc.returncode, "log:", log_path)
    if proc.returncode != 0:
        !tail -n 40 {log_path}
    return proc.returncode

COMMON_ESC = (
    f"--split-dir experiment_splits/esc_primary_slither "
    f"--contract-root {REPO} "
    f"--contracts-dir {ESC_CONTRACTS} "
    f"--sample-strategy reservoir --seed 42 "
    f"--max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 "
    f"--default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.0 "
)

COMMON_SB = (
    f"--split-dir experiment_splits/smartbugs_secondary_slither "
    f"--contract-root {REPO} "
    f"--contracts-dir {SB_CONTRACTS} "
    f"--sample-strategy reservoir --seed 42 "
)

THRESH_FINE = (
    "--threshold-candidates 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8"
)
```

---

## Cell 6 — ESC primary (full scale)

Run **one model per session** if Colab disconnects; after each run, copy `experiments/.../<run_name>/` to Drive.

### 6a — Slither detector baseline (no training, slow on 10k test)

```python
run_train(
    "python train_experiment.py --model slither "
    + COMMON_ESC
    + "--output-dir experiments/slither_baseline "
    + "--run-name esc_slither_slitherlabels_10k "
    + "--max-train-samples 0 --max-val-samples 2000 --max-test-samples 10000",
    "esc_slither",
)
```

### 6b — Ablation A: CodeBERT (semantic only)

```python
run_train(
    "python train_experiment.py --model codebert "
    + "--codebert-model-name microsoft/codebert-base "
    + COMMON_ESC
    + "--output-dir experiments/codebert_baseline "
    + "--run-name esc_codebert_slither_100k "
    + "--epochs 4 --train-batch-size 8 --eval-batch-size 8 --max-length 192 "
    + "--learning-rate 1.5e-5 --weight-decay 0.01 --max-pos-weight 8 "
    + "--grad-clip-norm 1.0 --threshold-min-precision 0.10 "
    + THRESH_FINE + " --save-model",
    "esc_codebert",
)
```

### 6c — Ablation B: GCN (structural only)

```python
run_train(
    "python train_experiment.py --model gnn "
    + COMMON_ESC
    + "--output-dir experiments/gnn_baseline "
    + "--run-name esc_gnn_slither_100k "
    + "--gnn-epochs 3 --gnn-max-nodes 48 --gnn-feature-dim 256 "
    + "--gnn-hidden-dim 128 --gnn-num-layers 2 --gnn-dropout 0.2 "
    + "--gnn-train-batch-size 64 --gnn-eval-batch-size 128 "
    + "--gnn-learning-rate 1e-3 --gnn-weight-decay 1e-4 "
    + "--gnn-max-pos-weight 8 --gnn-grad-clip-norm 1.0 "
    + THRESH_FINE,
    "esc_gnn",
)
```

### 6d — Ablation C: Hybrid + cross-modal attention (default hybrid)

```python
run_train(
    "python train_experiment.py --model hybrid "
    + "--codebert-model-name microsoft/codebert-base "
    + COMMON_ESC
    + "--output-dir experiments/hybrid_baseline "
    + "--run-name esc_hybrid_slither_100k "
    + "--hybrid-epochs 5 --hybrid-train-batch-size 2 --hybrid-eval-batch-size 4 "
    + "--max-length 192 --hybrid-max-nodes 96 --hybrid-feature-dim 256 "
    + "--hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 "
    + "--hybrid-fusion-dim 256 --hybrid-attention-heads 4 "
    + "--hybrid-graph-residual-scale 0.10 --hybrid-dropout 0.15 "
    + "--hybrid-transformer-learning-rate 1.5e-5 --hybrid-head-learning-rate 4e-4 "
    + "--hybrid-weight-decay 0.01 --hybrid-max-pos-weight 10 "
    + "--hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 "
    + "--hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric weighted_f1 "
    + THRESH_FINE + " --save-model",
    "esc_hybrid",
)
```

### 6e — Hybrid + cross-contract (C+)

```python
run_train(
    "python train_experiment.py --model hybrid "
    + "--codebert-model-name microsoft/codebert-base "
    + "--hybrid-enable-cross-contract "
    + COMMON_ESC
    + "--output-dir experiments/hybrid_baseline "
    + "--run-name esc_hybrid_crosscontract_slither_100k "
    + "--hybrid-epochs 5 --hybrid-train-batch-size 2 --hybrid-eval-batch-size 4 "
    + "--max-length 192 --hybrid-max-nodes 96 --hybrid-feature-dim 256 "
    + "--hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 "
    + "--hybrid-fusion-dim 256 --hybrid-attention-heads 4 "
    + "--hybrid-graph-residual-scale 0.10 --hybrid-dropout 0.15 "
    + "--hybrid-transformer-learning-rate 1.5e-5 --hybrid-head-learning-rate 4e-4 "
    + "--hybrid-weight-decay 0.01 --hybrid-max-pos-weight 10 "
    + "--hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 "
    + "--hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric weighted_f1 "
    + THRESH_FINE + " --save-model",
    "esc_hybrid_cross",
)
```

---

## Cell 7 — SmartBugs secondary (asymmetric scale)

| Model | Train cap | Why |
|-------|-----------|-----|
| CodeBERT | 100k | No graph build |
| GNN | 5k | Graph extraction expensive |
| Hybrid | 5k | Same |
| Slither | 10k test | Detector only |

### 7a — Slither baseline

```python
run_train(
    "python train_experiment.py --model slither "
    + COMMON_SB
    + "--output-dir experiments/slither_baseline "
    + "--run-name smartbugs_slither_slitherlabels_10k "
    + "--max-train-samples 0 --max-val-samples 500 --max-test-samples 10000",
    "sb_slither",
)
```

### 7b — CodeBERT 100k

```python
run_train(
    "python train_experiment.py --model codebert "
    + "--codebert-model-name microsoft/codebert-base "
    + COMMON_SB
    + "--max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 "
    + "--output-dir experiments/codebert_baseline "
    + "--run-name smartbugs_codebert_slither_100k "
    + "--epochs 4 --train-batch-size 8 --eval-batch-size 8 --max-length 192 "
    + "--learning-rate 1.5e-5 --weight-decay 0.01 --max-pos-weight 8 "
    + "--grad-clip-norm 1.0 --threshold-min-support 5 --threshold-min-precision 0.10 "
    + "--threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 "
    + "--save-model",
    "sb_codebert",
)
```

### 7c — GNN 5k

```python
run_train(
    "python train_experiment.py --model gnn "
    + COMMON_SB
    + "--max-train-samples 5000 --max-val-samples 500 --max-test-samples 500 "
    + "--output-dir experiments/gnn_baseline "
    + "--run-name smartbugs_gnn_slither_5k "
    + "--gnn-epochs 3 --gnn-max-nodes 48 --gnn-feature-dim 256 "
    + "--gnn-hidden-dim 160 --gnn-num-layers 2 --gnn-dropout 0.10 "
    + "--gnn-train-batch-size 4 --gnn-eval-batch-size 8 "
    + "--gnn-learning-rate 7e-4 --gnn-weight-decay 1e-4 "
    + "--gnn-max-pos-weight 8 --gnn-grad-clip-norm 1.0 "
    + "--threshold-min-support 3 --threshold-min-precision 0.10 "
    + "--threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6",
    "sb_gnn",
)
```

### 7d — Hybrid + cross-modal 5k

```python
run_train(
    "python train_experiment.py --model hybrid "
    + "--codebert-model-name microsoft/codebert-base "
    + COMMON_SB
    + "--max-train-samples 5000 --max-val-samples 500 --max-test-samples 500 "
    + "--output-dir experiments/hybrid_baseline "
    + "--run-name smartbugs_hybrid_slither_5k "
    + "--hybrid-epochs 3 --hybrid-train-batch-size 1 --hybrid-eval-batch-size 2 "
    + "--max-length 128 --hybrid-max-nodes 48 --hybrid-feature-dim 256 "
    + "--hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 "
    + "--hybrid-fusion-dim 256 --hybrid-attention-heads 4 "
    + "--hybrid-graph-residual-scale 0.08 --hybrid-dropout 0.15 "
    + "--hybrid-transformer-learning-rate 1e-5 --hybrid-head-learning-rate 3e-4 "
    + "--hybrid-weight-decay 0.01 --hybrid-max-pos-weight 10 "
    + "--hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 "
    + "--hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric weighted_f1 "
    + "--threshold-min-support 3 --threshold-min-precision 0.0 "
    + "--threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 "
    + "--save-model",
    "sb_hybrid",
)
```

### 7e — Hybrid + cross-contract 5k

```python
run_train(
    "python train_experiment.py --model hybrid "
    + "--codebert-model-name microsoft/codebert-base "
    + "--hybrid-enable-cross-contract "
    + COMMON_SB
    + "--max-train-samples 5000 --max-val-samples 500 --max-test-samples 500 "
    + "--output-dir experiments/hybrid_baseline "
    + "--run-name smartbugs_hybrid_crosscontract_slither_5k "
    + "--hybrid-epochs 3 --hybrid-train-batch-size 1 --hybrid-eval-batch-size 2 "
    + "--max-length 128 --hybrid-max-nodes 48 --hybrid-feature-dim 256 "
    + "--hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 "
    + "--hybrid-fusion-dim 256 --hybrid-attention-heads 4 "
    + "--hybrid-graph-residual-scale 0.08 --hybrid-dropout 0.15 "
    + "--hybrid-transformer-learning-rate 1e-5 --hybrid-head-learning-rate 3e-4 "
    + "--hybrid-weight-decay 0.01 --hybrid-max-pos-weight 10 "
    + "--hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 "
    + "--hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric weighted_f1 "
    + "--threshold-min-support 3 --threshold-min-precision 0.0 "
    + "--threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 "
    + "--save-model",
    "sb_hybrid_cross",
)
```

---

## Cell 8 — Copy results to Drive

```python
import shutil

def archive_run(rel_path: str):
    src = REPO / rel_path
    if not src.is_dir():
        print("skip missing", src)
        return
    dst = RESULTS_DRIVE / rel_path.replace("/", "_")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("archived", dst)

RUNS = [
    "experiments/slither_baseline/esc_slither_slitherlabels_10k",
    "experiments/codebert_baseline/esc_codebert_slither_100k",
    "experiments/gnn_baseline/esc_gnn_slither_100k",
    "experiments/hybrid_baseline/esc_hybrid_slither_100k",
    "experiments/hybrid_baseline/esc_hybrid_crosscontract_slither_100k",
    "experiments/slither_baseline/smartbugs_slither_slitherlabels_10k",
    "experiments/codebert_baseline/smartbugs_codebert_slither_100k",
    "experiments/gnn_baseline/smartbugs_gnn_slither_5k",
    "experiments/hybrid_baseline/smartbugs_hybrid_slither_5k",
    "experiments/hybrid_baseline/smartbugs_hybrid_crosscontract_slither_5k",
]
for r in RUNS:
    archive_run(r)
```

Each run folder should contain: `summary.txt`, `test_metrics.json`, `val_metrics.json`, `thresholds.json`, `run_config.json`.

---

## Cell 9 — Quick metrics table

```python
import json
from pathlib import Path

rows = []
for p in sorted((REPO / "experiments").rglob("test_metrics.json")):
    if "slither" not in str(p) and "esc_" not in str(p) and "smartbugs_" not in str(p):
        continue
    m = json.loads(p.read_text(encoding="utf-8"))
    rows.append((p.parent.name, m.get("micro_f1"), m.get("subset_accuracy"), str(p.parent.parent.name)))

print(f"{'run':<45} {'micro_f1':>8} {'subset_acc':>10}  family")
for name, f1, acc, fam in sorted(rows, key=lambda x: x[0]):
    print(f"{name:<45} {f1:>8.4f} {acc:>10.4f}  {fam}")
```

---

## Cell 10 (optional) — SHAP on best hybrid

After hybrid run with `--save-model`:

```python
!python run_shap_explain.py \
  --model hybrid \
  --run-dir experiments/hybrid_baseline/esc_hybrid_slither_100k \
  --split-dir experiment_splits/esc_primary_slither \
  --contracts-dir {ESC_CONTRACTS} \
  --contract-root {REPO} \
  --max-samples 50 \
  --output-dir thesis_md/shap_esc_hybrid_slither
```

---

## Run order (recommended)

1. Cell 1–4 (setup + path check)  
2. ESC: Slither → CodeBERT → GNN → Hybrid → Hybrid+cross-contract  
3. SmartBugs: same order (smaller GNN/Hybrid caps)  
4. Cell 8–9 archive + table  

**Estimated wall time (T4):** ESC CodeBERT ~2–4 h; ESC GNN ~4–8 h; ESC Hybrid ~6–12 h each; SmartBugs CodeBERT ~3–5 h; SmartBugs GNN/Hybrid ~2–4 h each; Slither baselines vary (hours on 10k test).

---

## Experiment matrix (thesis Table 4.1)

| Dataset | Slither T0 | CodeBERT A | GCN B | Hybrid C (cross-modal) | Hybrid C+ (cross-contract) |
|---------|------------|------------|-------|------------------------|----------------------------|
| ESC | `esc_slither_slitherlabels_10k` | `esc_codebert_slither_100k` | `esc_gnn_slither_100k` | `esc_hybrid_slither_100k` | `esc_hybrid_crosscontract_slither_100k` |
| SmartBugs | `smartbugs_slither_slitherlabels_10k` | `smartbugs_codebert_slither_100k` | `smartbugs_gnn_slither_5k` | `smartbugs_hybrid_slither_5k` | `smartbugs_hybrid_crosscontract_slither_5k` |
