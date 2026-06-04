# Colab — Final CodeBERT Run (best shot, run once)

**Pull latest code first** (adds `--codebert-checkpoint-metric`):

```python
!cd /content/Ethereum_smart_contract_datast && git pull
# or re-clone if you never pulled updates
```

## One cell — ESC + SmartBugs (final run names)

```python
import os, subprocess, sys
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/thesis")
DATA  = DRIVE / "Ethereum_smart_contract_datast"
REPO  = Path("/content/Ethereum_smart_contract_datast/Ethereum_smart_contract_datast")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

for z in ["esc_primary_slither.zip", "smartbugs_secondary_slither.zip"]:
    zp = DATA / z
    if zp.is_file():
        subprocess.run(["unzip", "-q", "-o", str(zp), "-d", "experiment_splits"], cwd=REPO)

if (DATA / "codebert_base.zip").is_file():
    subprocess.run(["unzip", "-q", "-o", str(DATA / "codebert_base.zip"), "-d", str(REPO)], check=False)
MODEL = "hf_models/codebert-base" if (REPO / "hf_models/codebert-base/config.json").is_file() else "microsoft/codebert-base"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate", "scikit-learn", "ijson"], check=False)

THRESH = (
    "0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"
)
# Best-shot hyperparameters (Slither splits, 100k cap, checkpoint = val micro_f1)
COMMON = (
    f"--codebert-model-name {MODEL} "
    "--sample-strategy reservoir --epochs 5 "
    "--train-batch-size 8 --eval-batch-size 16 --max-length 128 "
    "--learning-rate 2e-5 --weight-decay 0.01 --max-pos-weight 8 "
    "--grad-clip-norm 1.0 --device cuda --save-model "
    "--codebert-checkpoint-metric micro_f1 "
    "--default-threshold 0.5 --threshold-min-support 5 "
    "--threshold-min-precision 0.0 "
    f"--threshold-candidates {THRESH}"
)

def run(split, name, test_support=5):
    cmd = (
        f"python train_experiment.py --model codebert --split-dir {split} "
        f"--output-dir experiments/codebert_baseline --run-name {name} "
        f"--max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 "
        f"{COMMON}"
    )
    if test_support == 3:
        cmd = cmd.replace("--threshold-min-support 5", "--threshold-min-support 3")
    print("RUN", name)
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)

run("experiment_splits/esc_primary_slither", "esc_codebert_slither_final")
run("experiment_splits/smartbugs_secondary_slither", "smartbugs_codebert_slither_final", test_support=3)

print("DONE — use these run folders in thesis:")
print("  experiments/codebert_baseline/esc_codebert_slither_final")
print("  experiments/codebert_baseline/smartbugs_codebert_slither_final")
```

## Why this is the best single CodeBERT config

| Setting | Value | Reason |
|---------|--------|--------|
| `epochs` | 5 | Enough training; checkpoint picks best epoch by F1 |
| `codebert-checkpoint-metric` | `micro_f1` | Avoids epoch-4 val_loss overfit |
| `max_length` | 128 | Matches best historical 100k ESC run |
| `learning_rate` | 2e-5 | Proven for CodeBERT fine-tune |
| `max_pos_weight` | 8 | Balanced rare-label handling |
| `threshold-min-precision` | 0.0 | Lets Slither-sparse labels use lower thresholds |
| `threshold-candidates` | 0.1–0.9 | Full grid |

## Realistic expectations (Slither labels)

| Dataset | Old heuristic ~F1 | Slither final (typical) |
|---------|-------------------|-------------------------|
| ESC | ~0.93 | **~0.82–0.88** |
| SmartBugs | ~0.89 | **~0.70–0.80** |

You will **not** match 0.93 on Slither-only labels; that is a label change, not a failed tune.

## After run — copy to Drive

```python
import shutil
out = DRIVE / "codebert_slither_final"
out.mkdir(parents=True, exist_ok=True)
for n in ["esc_codebert_slither_final", "smartbugs_codebert_slither_final"]:
    shutil.copytree(REPO / "experiments/codebert_baseline" / n, out / n, dirs_exist_ok=True)
```
