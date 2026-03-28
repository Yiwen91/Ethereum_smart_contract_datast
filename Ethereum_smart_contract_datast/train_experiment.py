#!/usr/bin/env python3
"""
Train and evaluate experiment baselines on ESC split JSON files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiment_utils import (
    VULN_TYPES,
    LoadedSplit,
    apply_thresholds,
    choose_thresholds,
    compute_multilabel_metrics,
    load_named_split,
    metrics_to_text,
    save_json,
    save_predictions_jsonl,
)
from models_codebert import CodeBERTMultilabelBaseline
from models_tabular import TabularMultilabelBaseline


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_run_dir(base_dir: str | Path, run_name: str | None) -> Path:
    root = Path(base_dir)
    if run_name:
        run_dir = root / run_name
    else:
        run_dir = root / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _split_path(split_dir: str | Path, name: str) -> Path:
    return Path(split_dir) / f"{name}.json"


def _load_splits(args) -> tuple[LoadedSplit, LoadedSplit, LoadedSplit]:
    print("[load] Reading train split...")
    train_split = load_named_split(
        "train",
        _split_path(args.split_dir, "train"),
        max_samples=args.max_train_samples,
        seed=args.seed,
        sample_strategy=args.sample_strategy,
    )
    print("[load] Reading validation split...")
    val_split = load_named_split(
        "val",
        _split_path(args.split_dir, "val"),
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
        sample_strategy=args.sample_strategy,
    )
    print("[load] Reading test split...")
    test_split = load_named_split(
        "test",
        _split_path(args.split_dir, "test"),
        max_samples=args.max_test_samples,
        seed=args.seed + 2,
        sample_strategy=args.sample_strategy,
    )
    return train_split, val_split, test_split


def _config_from_args(args) -> dict:
    config = {
        "model": args.model,
        "split_dir": str(Path(args.split_dir).resolve()),
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "sample_strategy": args.sample_strategy,
    }
    if args.model == "tabular":
        config.update(
            {
                "max_features": args.max_features,
                "min_df": args.min_df,
                "max_df": args.max_df,
                "c_value": args.c_value,
                "max_iter": args.max_iter,
            }
        )
    elif args.model == "codebert":
        config.update(
            {
                "codebert_model_name": args.codebert_model_name,
                "max_length": args.max_length,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "device": args.device,
                "save_model": args.save_model,
            }
        )
    return config


def _save_run_outputs(
    *,
    run_dir: Path,
    summary_title: str,
    train_split: LoadedSplit,
    val_split: LoadedSplit,
    test_split: LoadedSplit,
    val_prob,
    val_pred,
    val_metrics,
    test_prob,
    test_pred,
    test_metrics,
    thresholds,
    config: dict,
):
    config["thresholds"] = thresholds
    config["splits"] = {
        "train_samples": len(train_split.records),
        "val_samples": len(val_split.records),
        "test_samples": len(test_split.records),
    }

    save_json(run_dir / "run_config.json", config)
    save_json(run_dir / "val_metrics.json", val_metrics)
    save_json(run_dir / "test_metrics.json", test_metrics)
    save_json(run_dir / "thresholds.json", thresholds)

    val_predictions = run_dir / "val_predictions.jsonl"
    test_predictions = run_dir / "test_predictions.jsonl"
    save_predictions_jsonl(
        val_predictions,
        val_split.records,
        val_split.labels,
        val_prob,
        val_pred,
        label_order=VULN_TYPES,
    )
    save_predictions_jsonl(
        test_predictions,
        test_split.records,
        test_split.labels,
        test_prob,
        test_pred,
        label_order=VULN_TYPES,
    )

    summary = "\n".join(
        [
            summary_title,
            "=" * 72,
            f"Run directory: {run_dir}",
            f"Train samples: {len(train_split.records)}",
            f"Val samples:   {len(val_split.records)}",
            f"Test samples:  {len(test_split.records)}",
            "",
            metrics_to_text("Validation", val_metrics, thresholds).rstrip(),
            "",
            metrics_to_text("Test", test_metrics, thresholds).rstrip(),
            "",
        ]
    )
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def run_tabular_experiment(args) -> Path:
    run_dir = _ensure_run_dir(args.output_dir, args.run_name)
    train_split, val_split, test_split = _load_splits(args)

    print("[train] Fitting TF-IDF + logistic regression baseline...")
    model = TabularMultilabelBaseline(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        c_value=args.c_value,
        max_iter=args.max_iter,
    )
    model.fit(train_split.texts, train_split.labels)

    print("[eval] Selecting per-label thresholds on validation split...")
    val_prob = model.predict_proba(val_split.texts)
    thresholds = choose_thresholds(val_split.labels, val_prob, label_order=VULN_TYPES)
    val_pred = apply_thresholds(val_prob, thresholds, label_order=VULN_TYPES)
    val_metrics = compute_multilabel_metrics(val_split.labels, val_pred, label_order=VULN_TYPES)

    print("[eval] Evaluating on test split...")
    test_prob = model.predict_proba(test_split.texts)
    test_pred = apply_thresholds(test_prob, thresholds, label_order=VULN_TYPES)
    test_metrics = compute_multilabel_metrics(test_split.labels, test_pred, label_order=VULN_TYPES)

    _save_run_outputs(
        run_dir=run_dir,
        summary_title="ESC Tabular Baseline Summary",
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        val_prob=val_prob,
        val_pred=val_pred,
        val_metrics=val_metrics,
        test_prob=test_prob,
        test_pred=test_pred,
        test_metrics=test_metrics,
        thresholds=thresholds,
        config=_config_from_args(args),
    )
    return run_dir


def run_codebert_experiment(args) -> Path:
    run_dir = _ensure_run_dir(args.output_dir, args.run_name)
    train_split, val_split, test_split = _load_splits(args)

    print("[train] Fine-tuning CodeBERT baseline...")
    model = CodeBERTMultilabelBaseline(
        model_name=args.codebert_model_name,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )
    model.fit(
        train_split.texts,
        train_split.labels,
        val_texts=val_split.texts,
        val_labels=val_split.labels,
    )

    print("[eval] Selecting per-label thresholds on validation split...")
    val_prob = model.predict_proba(val_split.texts)
    thresholds = choose_thresholds(val_split.labels, val_prob, label_order=VULN_TYPES)
    val_pred = apply_thresholds(val_prob, thresholds, label_order=VULN_TYPES)
    val_metrics = compute_multilabel_metrics(val_split.labels, val_pred, label_order=VULN_TYPES)

    print("[eval] Evaluating on test split...")
    test_prob = model.predict_proba(test_split.texts)
    test_pred = apply_thresholds(test_prob, thresholds, label_order=VULN_TYPES)
    test_metrics = compute_multilabel_metrics(test_split.labels, test_pred, label_order=VULN_TYPES)

    if args.save_model:
        model.save_model(str(run_dir / "model"))

    _save_run_outputs(
        run_dir=run_dir,
        summary_title="ESC CodeBERT Baseline Summary",
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        val_prob=val_prob,
        val_pred=val_pred,
        val_metrics=val_metrics,
        test_prob=test_prob,
        test_pred=test_pred,
        test_metrics=test_metrics,
        thresholds=thresholds,
        config=_config_from_args(args),
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline experiments on ESC function-level splits."
    )
    parser.add_argument(
        "--model",
        choices=["tabular", "codebert"],
        default="tabular",
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--split-dir",
        default="experiment_splits/esc_primary",
        help="Directory containing train.json, val.json, and test.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store run outputs.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional fixed run folder name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=100000)
    parser.add_argument("--max-val-samples", type=int, default=20000)
    parser.add_argument("--max-test-samples", type=int, default=20000)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--c-value", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--codebert-model-name", default="microsoft/codebert-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", help="Optional torch device override, e.g. cpu or cuda")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained Hugging Face model and tokenizer for codebert runs.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["reservoir", "head"],
        default="reservoir",
        help="How to sample examples when max sample limits are set.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = (
            "experiments/codebert_baseline"
            if args.model == "codebert"
            else "experiments/tabular_baseline"
        )
    if args.model == "tabular":
        run_tabular_experiment(args)
        return
    if args.model == "codebert":
        run_codebert_experiment(args)
        return
    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
