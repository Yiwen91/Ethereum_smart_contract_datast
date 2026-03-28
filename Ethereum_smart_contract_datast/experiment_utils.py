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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
) -> dict[str, float]:
    label_order = label_order or list(VULN_TYPES)
    candidate_thresholds = candidate_thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    thresholds: dict[str, float] = {}

    for idx, label in enumerate(label_order):
        best_threshold = 0.5
        best_f1 = -1.0
        y_true_col = y_true[:, idx]
        y_prob_col = y_prob[:, idx]
        for threshold in candidate_thresholds:
            y_pred_col = (y_prob_col >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true_col,
                y_pred_col,
                average="binary",
                zero_division=0,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[label] = best_threshold

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
    label_order: list[str] | None = None,
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
    per_label_p, per_label_r, per_label_f1, per_label_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label = {}
    for idx, label in enumerate(label_order):
        per_label[label] = {
            "precision": float(per_label_p[idx]),
            "recall": float(per_label_r[idx]),
            "f1": float(per_label_f1[idx]),
            "support": int(per_label_support[idx]),
            "predicted_positive": int(y_pred[:, idx].sum()),
        }

    exact_match = float(accuracy_score(y_true, y_pred))
    return {
        "samples": int(y_true.shape[0]),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "subset_accuracy": exact_match,
        "per_label": per_label,
    }


def metrics_to_text(name: str, metrics: dict, thresholds: dict[str, float] | None = None) -> str:
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
        f"Subset Accuracy: {metrics['subset_accuracy']:.4f}",
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
            f"{label:30} P={item['precision']:.4f} R={item['recall']:.4f} "
            f"F1={item['f1']:.4f} support={item['support']}{threshold_suffix}"
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

