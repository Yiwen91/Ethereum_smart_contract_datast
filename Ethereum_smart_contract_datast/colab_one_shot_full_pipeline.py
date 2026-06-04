#!/usr/bin/env python3
"""
Colab one-shot pipeline: all Slither-label experiments + SHAP.

Edit PATHS below, then in Colab:
  1) Runtime -> GPU (T4+), High-RAM if available
  2) Mount Drive, run setup cell from colab_one_shot_full_pipeline.md
  3) %run colab_one_shot_full_pipeline.py

Expected wall time on T4: ~20-40 hours for full ESC 100k + SmartBugs + SHAP.
Use Colab Pro / keep session alive. Set FAST_MODE=True only for wiring tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# EDIT THESE (Colab Drive layout)
# =============================================================================
DRIVE = Path("/content/drive/MyDrive/thesis")
# Code on fast local disk (set by colab_setup_cell.py); fallback for standalone run:
REPO = Path("/content/Ethereum_smart_contract_datast")
if not (REPO / "train_experiment.py").is_file():
    nested = REPO / "Ethereum_smart_contract_datast"
    if (nested / "train_experiment.py").is_file():
        REPO = nested
    else:
        REPO = DRIVE / "Ethereum_smart_contract_datast"


def _discover_contracts(drive: Path, marker: str) -> Path:
    candidates = [drive / marker, drive / "Ethereum_smart_contract_datast" / marker]
    if marker == "contract_dataset_ethereum":
        candidates.extend(p for p in drive.rglob("contract_dataset_ethereum") if p.is_dir())
    else:
        for hit in drive.rglob("smartbugs_wild"):
            c = hit / "contracts"
            if c.is_dir():
                candidates.append(c)
    best, best_n, seen = None, 0, set()
    for c in candidates:
        key = str(Path(c).resolve())
        if key in seen or not Path(c).is_dir():
            continue
        seen.add(key)
        n = sum(1 for _ in Path(c).rglob("*.sol"))
        if n > best_n:
            best_n, best = n, Path(c).resolve()
    if best is None:
        raise FileNotFoundError(f"Missing {marker} under {drive}")
    return best


ESC_CONTRACTS = _discover_contracts(DRIVE, "contract_dataset_ethereum")
SB_CONTRACTS = _discover_contracts(DRIVE, "smartbugs_wild/contracts")
RESULTS_DRIVE = DRIVE / "experiment_results_slither_one_shot"

# Use offline weights if you unzipped codebert_base.zip; else Hugging Face download
CODEBERT = "hf_models/codebert-base"
if not (REPO / CODEBERT / "config.json").is_file():
    CODEBERT = "microsoft/codebert-base"

FAST_MODE = False  # True = tiny samples (smoke); False = tuned production caps

ESC_SPLIT = "experiment_splits/esc_primary_slither"
SB_SPLIT = "experiment_splits/smartbugs_secondary_slither"

THRESH_FINE = (
    "0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8"
)
THRESH_SB = "0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7"

# =============================================================================
# Sample caps (FAST_MODE overrides)
# =============================================================================
if FAST_MODE:
    ESC_TRAIN, ESC_VAL, ESC_TEST = 2000, 400, 400
    SB_TRAIN, SB_VAL, SB_TEST = 1000, 200, 200
    SLITHER_TEST = 200
    HYBRID_EPOCHS_ESC, HYBRID_EPOCHS_SB = 1, 1
    CODEBERT_EPOCHS_ESC, CODEBERT_EPOCHS_SB = 1, 1
    GNN_EPOCHS = 1
    SHAP_SAMPLES, SHAP_EVALS = 3, 50
else:
    ESC_TRAIN, ESC_VAL, ESC_TEST = 100000, 10000, 10000
    SB_TRAIN, SB_VAL, SB_TEST = 100000, 10000, 10000
    SLITHER_TEST = 5000
    HYBRID_EPOCHS_ESC, HYBRID_EPOCHS_SB = 5, 3
    CODEBERT_EPOCHS_ESC, CODEBERT_EPOCHS_SB = 4, 4
    GNN_EPOCHS = 3
    SHAP_SAMPLES, SHAP_EVALS = 8, 100

SB_GNN_TRAIN, SB_GNN_VAL, SB_GNN_TEST = (
    (5000, 500, 500) if not FAST_MODE else (1000, 200, 200)
)


def _py() -> str:
    return sys.executable


def _common_esc() -> str:
    return (
        f"--split-dir {ESC_SPLIT} "
        f"--contract-root {REPO} "
        f"--contracts-dir {ESC_CONTRACTS} "
        f"--sample-strategy reservoir --seed 42 "
        f"--max-train-samples {ESC_TRAIN} --max-val-samples {ESC_VAL} "
        f"--max-test-samples {ESC_TEST} "
        f"--default-threshold 0.5 --threshold-min-support 5 "
        f"--threshold-min-precision 0.0 "
    )


def _common_sb() -> str:
    return (
        f"--split-dir {SB_SPLIT} "
        f"--contract-root {REPO} "
        f"--contracts-dir {SB_CONTRACTS} "
        f"--sample-strategy reservoir --seed 42 "
    )


def _run(name: str, cmd: str, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = datetime.now()
    print("\n" + "=" * 72)
    print(f"[{started.isoformat(timespec='seconds')}] START {name}")
    print(cmd)
    print("=" * 72, flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# {name}\n# {cmd}\n\n")
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(REPO),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = datetime.now() - started
    print(f"END {name} exit={proc.returncode} elapsed={elapsed}", flush=True)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        print("--- log tail ---")
        print("\n".join(tail))
    return proc.returncode


def _archive_run(rel: str, dest_root: Path) -> None:
    src = REPO / rel
    if not src.is_dir():
        print(f"[archive] skip missing {src}")
        return
    dst = dest_root / rel.replace("/", "__")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[archive] {dst}")


def _build_jobs() -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []

    # --- ESC ---
    jobs.append(
        (
            "esc_slither",
            f"{_py()} train_experiment.py --model slither {_common_esc()} "
            f"--max-train-samples 0 --max-val-samples 1000 --max-test-samples {SLITHER_TEST} "
            f"--output-dir experiments/slither_baseline "
            f"--run-name esc_slither_one_shot",
        )
    )
    jobs.append(
        (
            "esc_codebert",
            f"{_py()} train_experiment.py --model codebert "
            f"--codebert-model-name {CODEBERT} {_common_esc()} "
            f"--output-dir experiments/codebert_baseline "
            f"--run-name esc_codebert_one_shot "
            f"--epochs {CODEBERT_EPOCHS_ESC} --train-batch-size 8 --eval-batch-size 8 "
            f"--max-length 192 --learning-rate 1.5e-5 --weight-decay 0.01 "
            f"--max-pos-weight 8 --grad-clip-norm 1.0 --device cuda --save-model "
            f"--threshold-candidates {THRESH_FINE}",
        )
    )
    jobs.append(
        (
            "esc_gnn",
            f"{_py()} train_experiment.py --model gnn {_common_esc()} "
            f"--output-dir experiments/gnn_baseline --run-name esc_gnn_one_shot "
            f"--gnn-epochs {GNN_EPOCHS} --gnn-max-nodes 48 --gnn-feature-dim 256 "
            f"--gnn-hidden-dim 128 --gnn-num-layers 2 --gnn-dropout 0.2 "
            f"--gnn-train-batch-size 64 --gnn-eval-batch-size 128 "
            f"--gnn-learning-rate 1e-3 --gnn-weight-decay 1e-4 "
            f"--gnn-max-pos-weight 8 --gnn-grad-clip-norm 1.0 --device cuda "
            f"--threshold-candidates {THRESH_FINE}",
        )
    )
    jobs.append(
        (
            "esc_hybrid_crossmodal",
            f"{_py()} train_experiment.py --model hybrid "
            f"--codebert-model-name {CODEBERT} {_common_esc()} "
            f"--output-dir experiments/hybrid_baseline --run-name esc_hybrid_one_shot "
            f"--hybrid-epochs {HYBRID_EPOCHS_ESC} --hybrid-train-batch-size 2 "
            f"--hybrid-eval-batch-size 4 --max-length 192 --hybrid-max-nodes 96 "
            f"--hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 "
            f"--hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 "
            f"--hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.10 "
            f"--hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1.5e-5 "
            f"--hybrid-head-learning-rate 4e-4 --hybrid-weight-decay 0.01 "
            f"--hybrid-max-pos-weight 10 --hybrid-grad-clip-norm 1.0 "
            f"--hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 "
            f"--hybrid-checkpoint-metric weighted_f1 --device cuda --save-model "
            f"--threshold-candidates {THRESH_FINE}",
        )
    )
    jobs.append(
        (
            "esc_hybrid_crosscontract",
            f"{_py()} train_experiment.py --model hybrid "
            f"--codebert-model-name {CODEBERT} --hybrid-enable-cross-contract "
            f"{_common_esc()} "
            f"--output-dir experiments/hybrid_baseline "
            f"--run-name esc_hybrid_crosscontract_one_shot "
            f"--hybrid-epochs {HYBRID_EPOCHS_ESC} --hybrid-train-batch-size 2 "
            f"--hybrid-eval-batch-size 4 --max-length 192 --hybrid-max-nodes 96 "
            f"--hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 "
            f"--hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 "
            f"--hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.10 "
            f"--hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1.5e-5 "
            f"--hybrid-head-learning-rate 4e-4 --hybrid-weight-decay 0.01 "
            f"--hybrid-max-pos-weight 10 --hybrid-grad-clip-norm 1.0 "
            f"--hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 "
            f"--hybrid-checkpoint-metric weighted_f1 --device cuda --save-model "
            f"--threshold-candidates {THRESH_FINE}",
        )
    )

    # --- SmartBugs ---
    sb_common = (
        f"{_common_sb()} "
        f"--default-threshold 0.5 --threshold-min-support 5 "
    )
    jobs.append(
        (
            "sb_slither",
            f"{_py()} train_experiment.py --model slither {sb_common} "
            f"--max-train-samples 0 --max-val-samples 500 --max-test-samples {SLITHER_TEST} "
            f"--output-dir experiments/slither_baseline --run-name smartbugs_slither_one_shot",
        )
    )
    jobs.append(
        (
            "sb_codebert",
            f"{_py()} train_experiment.py --model codebert "
            f"--codebert-model-name {CODEBERT} {sb_common} "
            f"--max-train-samples {SB_TRAIN} --max-val-samples {SB_VAL} "
            f"--max-test-samples {SB_TEST} "
            f"--output-dir experiments/codebert_baseline --run-name smartbugs_codebert_one_shot "
            f"--epochs {CODEBERT_EPOCHS_SB} --train-batch-size 8 --eval-batch-size 8 "
            f"--max-length 192 --learning-rate 1.5e-5 --weight-decay 0.01 "
            f"--max-pos-weight 8 --grad-clip-norm 1.0 --device cuda --save-model "
            f"--threshold-min-precision 0.10 --threshold-candidates {THRESH_SB}",
        )
    )
    jobs.append(
        (
            "sb_gnn",
            f"{_py()} train_experiment.py --model gnn {sb_common} "
            f"--max-train-samples {SB_GNN_TRAIN} --max-val-samples {SB_GNN_VAL} "
            f"--max-test-samples {SB_GNN_TEST} "
            f"--output-dir experiments/gnn_baseline --run-name smartbugs_gnn_one_shot "
            f"--gnn-epochs {GNN_EPOCHS} --gnn-max-nodes 48 --gnn-feature-dim 256 "
            f"--gnn-hidden-dim 160 --gnn-num-layers 2 --gnn-dropout 0.10 "
            f"--gnn-train-batch-size 4 --gnn-eval-batch-size 8 "
            f"--gnn-learning-rate 7e-4 --gnn-weight-decay 1e-4 "
            f"--gnn-max-pos-weight 8 --gnn-grad-clip-norm 1.0 --device cuda "
            f"--threshold-min-support 3 --threshold-min-precision 0.10 "
            f"--threshold-candidates 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6",
        )
    )
    jobs.append(
        (
            "sb_hybrid_crossmodal",
            f"{_py()} train_experiment.py --model hybrid "
            f"--codebert-model-name {CODEBERT} {sb_common} "
            f"--max-train-samples {SB_GNN_TRAIN} --max-val-samples {SB_GNN_VAL} "
            f"--max-test-samples {SB_GNN_TEST} "
            f"--output-dir experiments/hybrid_baseline --run-name smartbugs_hybrid_one_shot "
            f"--hybrid-epochs {HYBRID_EPOCHS_SB} --hybrid-train-batch-size 1 "
            f"--hybrid-eval-batch-size 2 --max-length 128 --hybrid-max-nodes 48 "
            f"--hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 "
            f"--hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 "
            f"--hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.08 "
            f"--hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1e-5 "
            f"--hybrid-head-learning-rate 3e-4 --hybrid-weight-decay 0.01 "
            f"--hybrid-max-pos-weight 10 --hybrid-grad-clip-norm 1.0 "
            f"--hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 "
            f"--hybrid-checkpoint-metric weighted_f1 --device cuda --save-model "
            f"--threshold-min-support 3 --threshold-min-precision 0.0 "
            f"--threshold-candidates {THRESH_SB}",
        )
    )
    jobs.append(
        (
            "sb_hybrid_crosscontract",
            f"{_py()} train_experiment.py --model hybrid "
            f"--codebert-model-name {CODEBERT} --hybrid-enable-cross-contract "
            f"{sb_common} "
            f"--max-train-samples {SB_GNN_TRAIN} --max-val-samples {SB_GNN_VAL} "
            f"--max-test-samples {SB_GNN_TEST} "
            f"--output-dir experiments/hybrid_baseline "
            f"--run-name smartbugs_hybrid_crosscontract_one_shot "
            f"--hybrid-epochs {HYBRID_EPOCHS_SB} --hybrid-train-batch-size 1 "
            f"--hybrid-eval-batch-size 2 --max-length 128 --hybrid-max-nodes 48 "
            f"--hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 "
            f"--hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 "
            f"--hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.08 "
            f"--hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1e-5 "
            f"--hybrid-head-learning-rate 3e-4 --hybrid-weight-decay 0.01 "
            f"--hybrid-max-pos-weight 10 --hybrid-grad-clip-norm 1.0 "
            f"--hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 "
            f"--hybrid-checkpoint-metric weighted_f1 --device cuda --save-model "
            f"--threshold-min-support 3 --threshold-min-precision 0.0 "
            f"--threshold-candidates {THRESH_SB}",
        )
    )
    return jobs


SHAP_LABELS = ["Reentrancy", "Transaction-Ordering Dependence"]


def _run_shap(log_dir: Path) -> None:
    hybrid_run = REPO / "experiments/hybrid_baseline/esc_hybrid_one_shot"
    if not (hybrid_run / "model" / "hybrid_state.pt").is_file():
        print("[shap] SKIP: train esc_hybrid_one_shot with --save-model first")
        return
    for label in SHAP_LABELS:
        safe = label.replace(" ", "_").lower()
        name = f"shap_esc_hybrid_{safe}"
        cmd = (
            f"{_py()} run_shap_explain.py --model hybrid "
            f"--run-dir experiments/hybrid_baseline/esc_hybrid_one_shot "
            f"--split-dir {ESC_SPLIT} --split val "
            f"--contract-root {REPO} --contracts-dir {ESC_CONTRACTS} "
            f"--label \"{label}\" --max-samples {SHAP_SAMPLES} "
            f"--background-samples 20 --max-evals {SHAP_EVALS} --device cuda "
            f"--output-dir thesis_md/shap_one_shot/esc_hybrid_{safe}"
        )
        _run(name, cmd, log_dir)


def _print_metrics_table() -> None:
    rows = []
    for path in sorted((REPO / "experiments").rglob("test_metrics.json")):
        if "one_shot" not in str(path):
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                path.parent.name,
                m.get("micro_f1"),
                m.get("weighted_f1"),
                m.get("subset_accuracy"),
            )
        )
    print("\n" + "=" * 72)
    print("ONE-SHOT TEST METRICS")
    print(f"{'run':<42} {'micro_f1':>9} {'w_f1':>9} {'subset':>9}")
    for name, f1, wf1, acc in sorted(rows, key=lambda r: r[0]):
        print(f"{name:<42} {f1 or 0:>9.4f} {wf1 or 0:>9.4f} {acc or 0:>9.4f}")
    print("=" * 72)


def main() -> int:
    os.chdir(REPO)
    RESULTS_DRIVE.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS_DRIVE / "logs"
    manifest: list[dict] = []

    print("REPO", REPO)
    print("FAST_MODE", FAST_MODE)
    print("CODEBERT", CODEBERT)

    failed = 0
    for name, cmd in _build_jobs():
        code = _run(name, cmd, log_dir)
        manifest.append({"name": name, "exit": code, "cmd": cmd})
        if code != 0:
            failed += 1
            print(f"[warn] {name} failed; continuing pipeline...")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    _run_shap(log_dir)

    archive_dirs = [
        "experiments/slither_baseline/esc_slither_one_shot",
        "experiments/codebert_baseline/esc_codebert_one_shot",
        "experiments/gnn_baseline/esc_gnn_one_shot",
        "experiments/hybrid_baseline/esc_hybrid_one_shot",
        "experiments/hybrid_baseline/esc_hybrid_crosscontract_one_shot",
        "experiments/slither_baseline/smartbugs_slither_one_shot",
        "experiments/codebert_baseline/smartbugs_codebert_one_shot",
        "experiments/gnn_baseline/smartbugs_gnn_one_shot",
        "experiments/hybrid_baseline/smartbugs_hybrid_one_shot",
        "experiments/hybrid_baseline/smartbugs_hybrid_crosscontract_one_shot",
    ]
    for rel in archive_dirs:
        _archive_run(rel, RESULTS_DRIVE)
    shap_src = REPO / "thesis_md/shap_one_shot"
    if shap_src.is_dir():
        _archive_run("thesis_md/shap_one_shot", RESULTS_DRIVE)

    (RESULTS_DRIVE / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _print_metrics_table()
    print(f"\nDone. failed_jobs={failed}. Results: {RESULTS_DRIVE}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
