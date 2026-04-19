#!/usr/bin/env python3
"""
Train and evaluate experiment baselines on ESC split JSON files.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

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
from models_gnn import ASTCFGFunctionGNNMultilabelBaseline
from models_hybrid import HybridCodeBERTGNNMultilabelBaseline
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
                "max_pos_weight": args.max_pos_weight,
                "grad_clip_norm": args.grad_clip_norm,
            }
        )
    elif args.model == "gnn":
        config.update(
            {
                "gnn_max_nodes": args.gnn_max_nodes,
                "gnn_feature_dim": args.gnn_feature_dim,
                "gnn_hidden_dim": args.gnn_hidden_dim,
                "gnn_num_layers": args.gnn_num_layers,
                "gnn_dropout": args.gnn_dropout,
                "gnn_train_batch_size": args.gnn_train_batch_size,
                "gnn_eval_batch_size": args.gnn_eval_batch_size,
                "gnn_learning_rate": args.gnn_learning_rate,
                "gnn_weight_decay": args.gnn_weight_decay,
                "gnn_epochs": args.gnn_epochs,
                "gnn_max_pos_weight": args.gnn_max_pos_weight,
                "gnn_grad_clip_norm": args.gnn_grad_clip_norm,
                "device": args.device,
            }
        )
    elif args.model == "hybrid":
        config.update(
            {
                "codebert_model_name": args.codebert_model_name,
                "max_length": args.max_length,
                "hybrid_max_nodes": args.hybrid_max_nodes,
                "hybrid_feature_dim": args.hybrid_feature_dim,
                "hybrid_graph_hidden_dim": args.hybrid_graph_hidden_dim,
                "hybrid_graph_num_layers": args.hybrid_graph_num_layers,
                "hybrid_fusion_dim": args.hybrid_fusion_dim,
                "hybrid_attention_heads": args.hybrid_attention_heads,
                "hybrid_dropout": args.hybrid_dropout,
                "hybrid_train_batch_size": args.hybrid_train_batch_size,
                "hybrid_eval_batch_size": args.hybrid_eval_batch_size,
                "hybrid_transformer_learning_rate": args.hybrid_transformer_learning_rate,
                "hybrid_head_learning_rate": args.hybrid_head_learning_rate,
                "hybrid_weight_decay": args.hybrid_weight_decay,
                "hybrid_epochs": args.hybrid_epochs,
                "hybrid_max_pos_weight": args.hybrid_max_pos_weight,
                "hybrid_grad_clip_norm": args.hybrid_grad_clip_norm,
                "device": args.device,
                "save_model": args.save_model,
            }
        )
    config.update(
        {
            "default_threshold": args.default_threshold,
            "threshold_min_support": args.threshold_min_support,
            "threshold_min_precision": args.threshold_min_precision,
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
    return {
        "run_dir": str(run_dir),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "thresholds": thresholds,
    }


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
    val_start = perf_counter()
    val_prob = model.predict_proba(val_split.texts)
    val_inference_seconds = perf_counter() - val_start
    thresholds = choose_thresholds(
        val_split.labels,
        val_prob,
        label_order=VULN_TYPES,
        default_threshold=args.default_threshold,
        min_support=args.threshold_min_support,
        min_precision=args.threshold_min_precision,
    )
    val_pred = apply_thresholds(val_prob, thresholds, label_order=VULN_TYPES)
    val_metrics = compute_multilabel_metrics(
        val_split.labels,
        val_pred,
        y_prob=val_prob,
        label_order=VULN_TYPES,
        inference_seconds=val_inference_seconds,
    )

    print("[eval] Evaluating on test split...")
    test_start = perf_counter()
    test_prob = model.predict_proba(test_split.texts)
    test_inference_seconds = perf_counter() - test_start
    test_pred = apply_thresholds(test_prob, thresholds, label_order=VULN_TYPES)
    test_metrics = compute_multilabel_metrics(
        test_split.labels,
        test_pred,
        y_prob=test_prob,
        label_order=VULN_TYPES,
        inference_seconds=test_inference_seconds,
    )

    result = _save_run_outputs(
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
    return result


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
        max_pos_weight=args.max_pos_weight,
        grad_clip_norm=args.grad_clip_norm,
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
    val_start = perf_counter()
    val_prob = model.predict_proba(val_split.texts)
    val_inference_seconds = perf_counter() - val_start
    thresholds = choose_thresholds(
        val_split.labels,
        val_prob,
        label_order=VULN_TYPES,
        default_threshold=args.default_threshold,
        min_support=args.threshold_min_support,
        min_precision=args.threshold_min_precision,
    )
    val_pred = apply_thresholds(val_prob, thresholds, label_order=VULN_TYPES)
    val_metrics = compute_multilabel_metrics(
        val_split.labels,
        val_pred,
        y_prob=val_prob,
        label_order=VULN_TYPES,
        inference_seconds=val_inference_seconds,
    )

    print("[eval] Evaluating on test split...")
    test_start = perf_counter()
    test_prob = model.predict_proba(test_split.texts)
    test_inference_seconds = perf_counter() - test_start
    test_pred = apply_thresholds(test_prob, thresholds, label_order=VULN_TYPES)
    test_metrics = compute_multilabel_metrics(
        test_split.labels,
        test_pred,
        y_prob=test_prob,
        label_order=VULN_TYPES,
        inference_seconds=test_inference_seconds,
    )

    if args.save_model:
        model.save_model(str(run_dir / "model"))

    result = _save_run_outputs(
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
    return result


def run_gnn_experiment(args) -> Path:
    run_dir = _ensure_run_dir(args.output_dir, args.run_name)
    train_split, val_split, test_split = _load_splits(args)

    print("[train] Training AST/CFG GNN baseline...")
    model = ASTCFGFunctionGNNMultilabelBaseline(
        max_nodes=args.gnn_max_nodes,
        feature_dim=args.gnn_feature_dim,
        hidden_dim=args.gnn_hidden_dim,
        num_layers=args.gnn_num_layers,
        dropout=args.gnn_dropout,
        train_batch_size=args.gnn_train_batch_size,
        eval_batch_size=args.gnn_eval_batch_size,
        learning_rate=args.gnn_learning_rate,
        weight_decay=args.gnn_weight_decay,
        epochs=args.gnn_epochs,
        max_pos_weight=args.gnn_max_pos_weight,
        grad_clip_norm=args.gnn_grad_clip_norm,
        device=args.device,
        seed=args.seed,
    )
    model.fit(
        train_split.records,
        train_split.labels,
        val_records=val_split.records,
        val_labels=val_split.labels,
    )

    print("[eval] Selecting per-label thresholds on validation split...")
    val_start = perf_counter()
    val_prob = model.predict_proba(val_split.records)
    val_inference_seconds = perf_counter() - val_start
    thresholds = choose_thresholds(
        val_split.labels,
        val_prob,
        label_order=VULN_TYPES,
        default_threshold=args.default_threshold,
        min_support=args.threshold_min_support,
        min_precision=args.threshold_min_precision,
    )
    val_pred = apply_thresholds(val_prob, thresholds, label_order=VULN_TYPES)
    val_metrics = compute_multilabel_metrics(
        val_split.labels,
        val_pred,
        y_prob=val_prob,
        label_order=VULN_TYPES,
        inference_seconds=val_inference_seconds,
    )

    print("[eval] Evaluating on test split...")
    test_start = perf_counter()
    test_prob = model.predict_proba(test_split.records)
    test_inference_seconds = perf_counter() - test_start
    test_pred = apply_thresholds(test_prob, thresholds, label_order=VULN_TYPES)
    test_metrics = compute_multilabel_metrics(
        test_split.labels,
        test_pred,
        y_prob=test_prob,
        label_order=VULN_TYPES,
        inference_seconds=test_inference_seconds,
    )

    result = _save_run_outputs(
        run_dir=run_dir,
        summary_title="ESC GNN Baseline Summary",
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
    return result


def run_hybrid_experiment(args) -> Path:
    run_dir = _ensure_run_dir(args.output_dir, args.run_name)
    train_split, val_split, test_split = _load_splits(args)

    print("[train] Training Hybrid CodeBERT + AST/CFG GNN model...")
    model = HybridCodeBERTGNNMultilabelBaseline(
        model_name=args.codebert_model_name,
        max_length=args.max_length,
        max_nodes=args.hybrid_max_nodes,
        feature_dim=args.hybrid_feature_dim,
        graph_hidden_dim=args.hybrid_graph_hidden_dim,
        graph_num_layers=args.hybrid_graph_num_layers,
        fusion_dim=args.hybrid_fusion_dim,
        attention_heads=args.hybrid_attention_heads,
        dropout=args.hybrid_dropout,
        train_batch_size=args.hybrid_train_batch_size,
        eval_batch_size=args.hybrid_eval_batch_size,
        transformer_learning_rate=args.hybrid_transformer_learning_rate,
        head_learning_rate=args.hybrid_head_learning_rate,
        weight_decay=args.hybrid_weight_decay,
        epochs=args.hybrid_epochs,
        max_pos_weight=args.hybrid_max_pos_weight,
        grad_clip_norm=args.hybrid_grad_clip_norm,
        device=args.device,
        seed=args.seed,
    )
    model.fit(
        train_split.records,
        train_split.labels,
        val_records=val_split.records,
        val_labels=val_split.labels,
    )

    print("[eval] Selecting per-label thresholds on validation split...")
    val_start = perf_counter()
    val_prob = model.predict_proba(val_split.records)
    val_inference_seconds = perf_counter() - val_start
    thresholds = choose_thresholds(
        val_split.labels,
        val_prob,
        label_order=VULN_TYPES,
        default_threshold=args.default_threshold,
        min_support=args.threshold_min_support,
        min_precision=args.threshold_min_precision,
    )
    val_pred = apply_thresholds(val_prob, thresholds, label_order=VULN_TYPES)
    val_metrics = compute_multilabel_metrics(
        val_split.labels,
        val_pred,
        y_prob=val_prob,
        label_order=VULN_TYPES,
        inference_seconds=val_inference_seconds,
    )

    print("[eval] Evaluating on test split...")
    test_start = perf_counter()
    test_prob = model.predict_proba(test_split.records)
    test_inference_seconds = perf_counter() - test_start
    test_pred = apply_thresholds(test_prob, thresholds, label_order=VULN_TYPES)
    test_metrics = compute_multilabel_metrics(
        test_split.labels,
        test_pred,
        y_prob=test_prob,
        label_order=VULN_TYPES,
        inference_seconds=test_inference_seconds,
    )

    if args.save_model:
        model.save_model(str(run_dir / "model"))

    result = _save_run_outputs(
        run_dir=run_dir,
        summary_title="ESC Hybrid CodeBERT + AST/CFG GNN Summary",
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
    return result


def _aggregate_metric_group(metric_dicts: list[dict]) -> dict:
    scalar_keys = [
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "micro_auc_roc",
        "macro_auc_roc",
        "weighted_auc_roc",
        "subset_accuracy",
        "inference_latency_seconds",
        "inference_latency_ms_per_sample",
    ]
    aggregated = {
        "runs": len(metric_dicts),
        "samples_per_run": [metrics["samples"] for metrics in metric_dicts],
        "scalars": {},
        "per_label": {},
    }
    for key in scalar_keys:
        values = np.asarray(
            [metrics[key] for metrics in metric_dicts if metrics[key] is not None],
            dtype=np.float64,
        )
        if values.size == 0:
            aggregated["scalars"][key] = {
                "mean": None,
                "std": None,
            }
            continue
        aggregated["scalars"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    for label in VULN_TYPES:
        label_accuracy = np.asarray(
            [metrics["per_label"][label]["accuracy"] for metrics in metric_dicts],
            dtype=np.float64,
        )
        label_f1 = np.asarray([metrics["per_label"][label]["f1"] for metrics in metric_dicts], dtype=np.float64)
        label_precision = np.asarray([metrics["per_label"][label]["precision"] for metrics in metric_dicts], dtype=np.float64)
        label_recall = np.asarray([metrics["per_label"][label]["recall"] for metrics in metric_dicts], dtype=np.float64)
        label_support = np.asarray([metrics["per_label"][label]["support"] for metrics in metric_dicts], dtype=np.float64)
        auc_values = [
            metrics["per_label"][label]["auc_roc"]
            for metrics in metric_dicts
            if metrics["per_label"][label]["auc_roc"] is not None
        ]
        label_auc_mean = None
        label_auc_std = None
        if auc_values:
            label_auc = np.asarray(auc_values, dtype=np.float64)
            label_auc_mean = float(label_auc.mean())
            label_auc_std = float(label_auc.std(ddof=0))
        aggregated["per_label"][label] = {
            "accuracy_mean": float(label_accuracy.mean()),
            "accuracy_std": float(label_accuracy.std(ddof=0)),
            "precision_mean": float(label_precision.mean()),
            "precision_std": float(label_precision.std(ddof=0)),
            "recall_mean": float(label_recall.mean()),
            "recall_std": float(label_recall.std(ddof=0)),
            "f1_mean": float(label_f1.mean()),
            "f1_std": float(label_f1.std(ddof=0)),
            "auc_roc_mean": label_auc_mean,
            "auc_roc_std": label_auc_std,
            "support_mean": float(label_support.mean()),
        }
    return aggregated


def _format_aggregate_summary(model_name: str, seeds: list[int], val_agg: dict, test_agg: dict) -> str:
    def _format_optional(value) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def _section(title: str, aggregate: dict) -> list[str]:
        lines = [
            title,
            "=" * 72,
        ]
        for key, stats in aggregate["scalars"].items():
            lines.append(
                f"{key}: mean={_format_optional(stats['mean'])} std={_format_optional(stats['std'])}"
            )
        lines.extend([
            "",
            "Per-label metrics mean/std",
            "-" * 72,
        ])
        for label in VULN_TYPES:
            item = aggregate["per_label"][label]
            lines.append(
                f"{label:30} Acc={item['accuracy_mean']:.4f}+/-{item['accuracy_std']:.4f} "
                f"F1={item['f1_mean']:.4f}+/-{item['f1_std']:.4f} "
                f"P={item['precision_mean']:.4f}+/-{item['precision_std']:.4f} "
                f"R={item['recall_mean']:.4f}+/-{item['recall_std']:.4f} "
                f"AUC={_format_optional(item['auc_roc_mean'])}+/-{_format_optional(item['auc_roc_std'])}"
            )
        return lines

    lines = [
        f"ESC {model_name} Multi-Seed Summary",
        "=" * 72,
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
    ]
    lines.extend(_section("Validation Aggregate", val_agg))
    lines.append("")
    lines.extend(_section("Test Aggregate", test_agg))
    return "\n".join(lines) + "\n"


def run_multi_seed_experiments(args):
    seeds = args.seeds if args.seeds else [args.seed]
    base_run_name = args.run_name or f"{args.model}_{_timestamp()}_multiseed"
    run_results = []

    for seed in seeds:
        seed_args = copy.deepcopy(args)
        seed_args.seed = seed
        seed_args.run_name = f"{base_run_name}_seed{seed}"
        print(f"[multi-seed] Starting seed {seed}...")
        if seed_args.model == "tabular":
            result = run_tabular_experiment(seed_args)
        elif seed_args.model == "codebert":
            result = run_codebert_experiment(seed_args)
        elif seed_args.model == "gnn":
            result = run_gnn_experiment(seed_args)
        elif seed_args.model == "hybrid":
            result = run_hybrid_experiment(seed_args)
        else:
            raise ValueError(f"Unsupported model: {seed_args.model}")
        run_results.append(
            {
                "seed": seed,
                "run_dir": result["run_dir"],
                "val_metrics": result["val_metrics"],
                "test_metrics": result["test_metrics"],
            }
        )

    val_agg = _aggregate_metric_group([result["val_metrics"] for result in run_results])
    test_agg = _aggregate_metric_group([result["test_metrics"] for result in run_results])

    aggregate_dir = Path(args.output_dir) / f"{base_run_name}_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "seeds": seeds,
        "runs": run_results,
        "validation": val_agg,
        "test": test_agg,
        "config_template": _config_from_args(args),
    }
    save_json(aggregate_dir / "aggregate_metrics.json", payload)
    summary = _format_aggregate_summary(args.model.upper(), seeds, val_agg, test_agg)
    (aggregate_dir / "aggregate_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return aggregate_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline experiments on ESC function-level splits."
    )
    parser.add_argument(
        "--model",
        choices=["tabular", "codebert", "gnn", "hybrid"],
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional list of seeds for repeated runs and mean/std aggregation.",
    )
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
    parser.add_argument("--max-pos-weight", type=float, default=8.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
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
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-min-support", type=int, default=5)
    parser.add_argument("--threshold-min-precision", type=float, default=0.15)
    parser.add_argument("--gnn-max-nodes", type=int, default=48)
    parser.add_argument("--gnn-feature-dim", type=int, default=256)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.2)
    parser.add_argument("--gnn-train-batch-size", type=int, default=64)
    parser.add_argument("--gnn-eval-batch-size", type=int, default=128)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gnn-epochs", type=int, default=3)
    parser.add_argument("--gnn-max-pos-weight", type=float, default=8.0)
    parser.add_argument("--gnn-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--hybrid-max-nodes", type=int, default=128)
    parser.add_argument("--hybrid-feature-dim", type=int, default=256)
    parser.add_argument("--hybrid-graph-hidden-dim", type=int, default=128)
    parser.add_argument("--hybrid-graph-num-layers", type=int, default=2)
    parser.add_argument("--hybrid-fusion-dim", type=int, default=256)
    parser.add_argument("--hybrid-attention-heads", type=int, default=4)
    parser.add_argument("--hybrid-dropout", type=float, default=0.2)
    parser.add_argument("--hybrid-train-batch-size", type=int, default=4)
    parser.add_argument("--hybrid-eval-batch-size", type=int, default=8)
    parser.add_argument("--hybrid-transformer-learning-rate", type=float, default=2e-5)
    parser.add_argument("--hybrid-head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--hybrid-weight-decay", type=float, default=0.01)
    parser.add_argument("--hybrid-epochs", type=int, default=3)
    parser.add_argument("--hybrid-max-pos-weight", type=float, default=8.0)
    parser.add_argument("--hybrid-grad-clip-norm", type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = (
            "experiments/codebert_baseline"
            if args.model == "codebert"
            else "experiments/gnn_baseline"
            if args.model == "gnn"
            else "experiments/hybrid_baseline"
            if args.model == "hybrid"
            else "experiments/tabular_baseline"
        )
    if args.seeds and len(args.seeds) > 1:
        run_multi_seed_experiments(args)
        return
    if args.seeds and len(args.seeds) == 1:
        args.seed = args.seeds[0]
    if args.model == "tabular":
        run_tabular_experiment(args)
        return
    if args.model == "codebert":
        run_codebert_experiment(args)
        return
    if args.model == "gnn":
        run_gnn_experiment(args)
        return
    if args.model == "hybrid":
        run_hybrid_experiment(args)
        return
    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
