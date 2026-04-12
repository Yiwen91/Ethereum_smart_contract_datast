#!/usr/bin/env python3
"""
Shared experiment utilities for function-level multilabel training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from report_vulnerability_counts import VULN_TYPES

try:
    import ijson
except ImportError:  # pragma: no cover - exercised only when dependency missing
    ijson = None


@dataclass
class LoadedSplit:
    name: str
    records: list[dict]
    texts: list[str]
    labels: np.ndarray


def load_split_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_vulnerabilities(vulnerabilities: Iterable[str], label_order: list[str] | None = None) -> list[int]:
    label_order = label_order or list(VULN_TYPES)
    present = set(vulnerabilities)
    return [1 if label in present else 0 for label in label_order]


def _iter_function_records(json_path: str | Path) -> Iterable[dict]:
    json_path = Path(json_path)
    if ijson is not None:
        with json_path.open("rb") as handle:
            yield from ijson.items(handle, "functions.item")
        return

    if json_path.stat().st_size > 50 * 1024 * 1024:
        raise RuntimeError(
            f"{json_path} is too large to load without ijson. "
            "Install dependencies from requirements.txt before running experiments."
        )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for record in payload.get("functions", []):
        yield record


def _reservoir_sample(records: Iterable[dict], max_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    for idx, record in enumerate(records):
        if idx < max_samples:
            sample.append(record)
            continue
        replace_at = rng.randint(0, idx)
        if replace_at < max_samples:
            sample[replace_at] = record
    return sample


def load_split_records(
    json_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
    sample_strategy: str = "reservoir",
) -> list[dict]:
    records = _iter_function_records(json_path)
    if max_samples is None:
        return list(records)
    if max_samples <= 0:
        return []
    if sample_strategy == "head":
        sample: list[dict] = []
        for idx, record in enumerate(records):
            if idx >= max_samples:
                break
            sample.append(record)
        return sample
    return _reservoir_sample(records, max_samples=max_samples, seed=seed)


def load_named_split(
    name: str,
    json_path: str | Path,
    label_order: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    sample_strategy: str = "reservoir",
) -> LoadedSplit:
    label_order = label_order or list(VULN_TYPES)
    records = load_split_records(
        json_path,
        max_samples=max_samples,
        seed=seed,
        sample_strategy=sample_strategy,
    )
    texts = [record.get("function_code", "") for record in records]
    labels = np.asarray(
        [encode_vulnerabilities(record.get("vulnerabilities", []), label_order) for record in records],
        dtype=np.int32,
    )
    return LoadedSplit(name=name, records=records, texts=texts, labels=labels)


def choose_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_order: list[str] | None = None,
    candidate_thresholds: list[float] | None = None,
    default_threshold: float = 0.5,
    min_support: int = 5,
    min_precision: float = 0.15,
) -> dict[str, float]:
    label_order = label_order or list(VULN_TYPES)
    candidate_thresholds = candidate_thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds: dict[str, float] = {}

    for idx, label in enumerate(label_order):
        support = int(y_true[:, idx].sum())
        if support < min_support:
            thresholds[label] = default_threshold
            continue

        best_threshold = default_threshold
        best_f1 = -1.0
        constrained_threshold = None
        constrained_f1 = -1.0
        y_true_col = y_true[:, idx]
        y_prob_col = y_prob[:, idx]
        for threshold in candidate_thresholds:
            y_pred_col = (y_prob_col >= threshold).astype(int)
            precision, _, f1, _ = precision_recall_fscore_support(
                y_true_col,
                y_pred_col,
                average="binary",
                zero_division=0,
            )
            predicted_positive = int(y_pred_col.sum())
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            if predicted_positive > 0 and precision >= min_precision and f1 > constrained_f1:
                constrained_f1 = f1
                constrained_threshold = threshold
        thresholds[label] = constrained_threshold if constrained_threshold is not None else best_threshold

    return thresholds


def apply_thresholds(
    y_prob: np.ndarray,
    thresholds: dict[str, float],
    label_order: list[str] | None = None,
) -> np.ndarray:
    label_order = label_order or list(VULN_TYPES)
    threshold_array = np.asarray([thresholds[label] for label in label_order], dtype=np.float32)
    return (y_prob >= threshold_array).astype(np.int32)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    label_order: list[str] | None = None,
    inference_seconds: float | None = None,
) -> dict:
    label_order = label_order or list(VULN_TYPES)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    per_label_p, per_label_r, per_label_f1, per_label_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label = {}
    per_label_auc: list[float | None] = []
    for idx, label in enumerate(label_order):
        label_accuracy = float(accuracy_score(y_true[:, idx], y_pred[:, idx]))
        label_auc = None
        if y_prob is not None:
            y_true_col = y_true[:, idx]
            if np.unique(y_true_col).size > 1:
                label_auc = float(roc_auc_score(y_true_col, y_prob[:, idx]))
        per_label_auc.append(label_auc)
        per_label[label] = {
            "accuracy": label_accuracy,
            "precision": float(per_label_p[idx]),
            "recall": float(per_label_r[idx]),
            "f1": float(per_label_f1[idx]),
            "auc_roc": label_auc,
            "support": int(per_label_support[idx]),
            "predicted_positive": int(y_pred[:, idx].sum()),
        }

    exact_match = float(accuracy_score(y_true, y_pred))
    micro_auc = None
    macro_auc = None
    weighted_auc = None
    if y_prob is not None:
        micro_auc = float(roc_auc_score(y_true.reshape(-1), y_prob.reshape(-1)))
        valid_auc_pairs = [
            (auc_value, int(per_label_support[idx]))
            for idx, auc_value in enumerate(per_label_auc)
            if auc_value is not None
        ]
        if valid_auc_pairs:
            auc_values = np.asarray([item[0] for item in valid_auc_pairs], dtype=np.float64)
            auc_supports = np.asarray([item[1] for item in valid_auc_pairs], dtype=np.float64)
            macro_auc = float(auc_values.mean())
            if auc_supports.sum() > 0:
                weighted_auc = float(np.average(auc_values, weights=auc_supports))
            else:
                weighted_auc = macro_auc

    latency_total_seconds = None
    latency_avg_ms_per_sample = None
    if inference_seconds is not None:
        latency_total_seconds = float(inference_seconds)
        latency_avg_ms_per_sample = float((inference_seconds * 1000.0) / max(y_true.shape[0], 1))
    return {
        "samples": int(y_true.shape[0]),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "micro_auc_roc": micro_auc,
        "macro_auc_roc": macro_auc,
        "weighted_auc_roc": weighted_auc,
        "subset_accuracy": exact_match,
        "inference_latency_seconds": latency_total_seconds,
        "inference_latency_ms_per_sample": latency_avg_ms_per_sample,
        "per_label": per_label,
    }


def metrics_to_text(name: str, metrics: dict, thresholds: dict[str, float] | None = None) -> str:
    def _format_metric(value) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        f"{name} Metrics",
        "=" * 72,
        f"Samples: {metrics['samples']}",
        f"Micro Precision: {metrics['micro_precision']:.4f}",
        f"Micro Recall:    {metrics['micro_recall']:.4f}",
        f"Micro F1:        {metrics['micro_f1']:.4f}",
        f"Macro Precision: {metrics['macro_precision']:.4f}",
        f"Macro Recall:    {metrics['macro_recall']:.4f}",
        f"Macro F1:        {metrics['macro_f1']:.4f}",
        f"Weighted Precision: {metrics['weighted_precision']:.4f}",
        f"Weighted Recall:    {metrics['weighted_recall']:.4f}",
        f"Weighted F1:        {metrics['weighted_f1']:.4f}",
        f"Micro AUC-ROC:     {_format_metric(metrics['micro_auc_roc'])}",
        f"Macro AUC-ROC:     {_format_metric(metrics['macro_auc_roc'])}",
        f"Weighted AUC-ROC:  {_format_metric(metrics['weighted_auc_roc'])}",
        f"Subset Accuracy: {metrics['subset_accuracy']:.4f}",
        f"Inference Latency (s): {metrics['inference_latency_seconds']:.4f}" if metrics["inference_latency_seconds"] is not None else "Inference Latency (s): n/a",
        f"Inference Latency (ms/sample): {metrics['inference_latency_ms_per_sample']:.4f}" if metrics["inference_latency_ms_per_sample"] is not None else "Inference Latency (ms/sample): n/a",
        "",
        "Per-label metrics",
        "-" * 72,
    ]
    for label in VULN_TYPES:
        item = metrics["per_label"][label]
        threshold_suffix = ""
        if thresholds is not None:
            threshold_suffix = f" threshold={thresholds[label]:.2f}"
        lines.append(
            f"{label:30} Acc={item['accuracy']:.4f} P={item['precision']:.4f} "
            f"R={item['recall']:.4f} F1={item['f1']:.4f} "
            f"AUC={_format_metric(item['auc_roc'])} support={item['support']}{threshold_suffix}"
        )
    return "\n".join(lines) + "\n"


def save_json(path: str | Path, payload: dict):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_predictions_jsonl(
    path: str | Path,
    records: list[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_order: list[str] | None = None,
):
    label_order = label_order or list(VULN_TYPES)
    out_path = Path(path)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            payload = {
                "contract_file": record.get("contract_file", ""),
                "contract_name": record.get("contract_name", ""),
                "function_name": record.get("function_name", ""),
                "function_signature": record.get("function_signature", ""),
                "true_labels": [label_order[j] for j, value in enumerate(y_true[idx]) if value],
                "predicted_labels": [label_order[j] for j, value in enumerate(y_pred[idx]) if value],
                "probabilities": {
                    label_order[j]: float(y_prob[idx, j]) for j in range(len(label_order))
                },
            }
            handle.write(json.dumps(payload) + "\n")

