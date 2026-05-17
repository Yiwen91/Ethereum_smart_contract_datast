#!/usr/bin/env python3
"""
SHAP-based interpretability for experiment models (tabular, CodeBERT, hybrid).

Hybrid explanations target the **semantic (text) branch** while holding the AST/CFG graph
fixed per function — practical for thesis case studies and sample-level reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiment_utils import VULN_TYPES

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    shap = None
    _SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    plt = None
    _MPL_AVAILABLE = False


def require_shap() -> None:
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install dependencies with: pip install shap matplotlib"
        )


@dataclass
class TokenAttribution:
    token: str
    shap_value: float


@dataclass
class FeatureAttribution:
    feature: str
    shap_value: float


@dataclass
class SampleShapExplanation:
    sample_index: int
    contract_file: str
    function_name: str
    label: str
    predicted_probability: float
    base_value: float
    attributions: list[TokenAttribution] | list[FeatureAttribution]
    explanation_type: str
    function_code_preview: str = ""


@dataclass
class ShapRunSummary:
    model: str
    run_dir: str
    label: str
    num_samples: int
    num_explained: int
    background_samples: int
    max_evals: int | None
    samples: list[SampleShapExplanation] = field(default_factory=list)
    top_global_tokens: list[dict[str, Any]] = field(default_factory=list)


def _top_attributions(
    items: list[TokenAttribution] | list[FeatureAttribution],
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: abs(item.shap_value), reverse=True)[:limit]
    if not ranked:
        return []
    if isinstance(ranked[0], TokenAttribution):
        return [{"token": item.token, "shap_value": float(item.shap_value)} for item in ranked]
    return [{"feature": item.feature, "shap_value": float(item.shap_value)} for item in ranked]


def _global_token_importance(samples: list[SampleShapExplanation], limit: int = 25) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sample in samples:
        if sample.explanation_type != "text_tokens":
            continue
        for item in sample.attributions:
            assert isinstance(item, TokenAttribution)
            scores[item.token] = scores.get(item.token, 0.0) + abs(item.shap_value)
            counts[item.token] = counts.get(item.token, 0) + 1
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]
    return [
        {
            "token": token,
            "mean_abs_shap": float(score / max(counts[token], 1)),
            "count": counts[token],
        }
        for token, score in ranked
    ]


def explain_tabular_label(
    model,
    *,
    texts: list[str],
    label: str,
    label_index: int,
    background_texts: list[str],
    max_evals: int | None = 500,
) -> list[SampleShapExplanation]:
    require_shap()
    from models_tabular import _ConstantLabelModel

    label_model = model.models[label_index]
    if isinstance(label_model, _ConstantLabelModel):
        return []

    background = model.transform(background_texts)
    if background.shape[0] == 0:
        raise ValueError("Background set is empty for tabular SHAP.")

    def predict_positive(features_matrix) -> np.ndarray:
        if hasattr(features_matrix, "toarray"):
            dense = features_matrix.toarray()
        else:
            dense = np.asarray(features_matrix)
        if dense.ndim == 1:
            dense = dense.reshape(1, -1)
        return label_model.predict_proba(dense)[:, 1]

    explainer = shap.Explainer(predict_positive, background, max_evals=max_evals)
    features = model.transform(texts)
    shap_values = explainer(features)

    if hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
        base_values = np.asarray(shap_values.base_values)
    else:
        values = np.asarray(shap_values)
        base_values = np.zeros(values.shape[0], dtype=np.float32)

    feature_names = np.asarray(model.vectorizer.get_feature_names_out())
    if values.ndim == 3:
        values = values[:, :, 1]
        base_values = base_values[:, 1] if base_values.ndim > 1 else base_values

    explanations: list[SampleShapExplanation] = []
    probs = model.predict_proba(texts)[:, label_index]
    for idx in range(len(texts)):
        row_values = values[idx]
        top_indices = np.argsort(np.abs(row_values))[-20:][::-1]
        attributions = [
            FeatureAttribution(feature=str(feature_names[j]), shap_value=float(row_values[j]))
            for j in top_indices
            if abs(float(row_values[j])) > 1e-8
        ]
        explanations.append(
            SampleShapExplanation(
                sample_index=idx,
                contract_file="",
                function_name="",
                label=label,
                predicted_probability=float(probs[idx]),
                base_value=float(base_values[idx]) if np.ndim(base_values) else float(base_values),
                attributions=attributions,
                explanation_type="tfidf_features",
                function_code_preview=texts[idx][:500],
            )
        )
    return explanations


def explain_text_model_label(
    *,
    predict_label_proba: Callable[[list[str]], np.ndarray],
    tokenizer,
    texts: list[str],
    records: list[dict],
    label: str,
    label_index: int,
    background_texts: list[str],
    max_evals: int | None = 100,
) -> list[SampleShapExplanation]:
    require_shap()

    masker = shap.maskers.Text(tokenizer, mask_token="...")
    explainer = shap.Explainer(
        lambda masked_texts: predict_label_proba(list(masked_texts)),
        masker,
        output_names=[label],
    )

    shap_values = explainer(texts, max_evals=max_evals)
    if hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
        base_values = np.asarray(shap_values.base_values)
    else:
        values = np.asarray(shap_values)
        base_values = np.zeros(len(texts), dtype=np.float32)

    probs = predict_label_proba(texts)
    explanations: list[SampleShapExplanation] = []
    for idx, text in enumerate(texts):
        token_attrs: list[TokenAttribution] = []
        if values.ndim == 3:
            row = values[idx, :, 0]
        else:
            row = values[idx]
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", 512),
            return_tensors=None,
        )
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        limit = min(len(tokens), len(row))
        for token_idx in range(limit):
            token_attrs.append(
                TokenAttribution(token=str(tokens[token_idx]), shap_value=float(row[token_idx]))
            )

        record = records[idx] if idx < len(records) else {}
        explanations.append(
            SampleShapExplanation(
                sample_index=idx,
                contract_file=str(record.get("contract_file", "")),
                function_name=str(record.get("function_name", "")),
                label=label,
                predicted_probability=float(probs[idx]),
                base_value=float(base_values[idx]) if np.ndim(base_values) else float(base_values),
                attributions=sorted(token_attrs, key=lambda item: abs(item.shap_value), reverse=True)[:30],
                explanation_type="text_tokens",
                function_code_preview=text[:500],
            )
        )
    return explanations


def build_hybrid_text_predict_fn(hybrid_model, record: dict, label_index: int):
    """Predict one label probability while varying text; graph + cross-contract fixed."""
    import torch

    assert hybrid_model.model is not None
    base_batch = hybrid_model._collate(
        [{"record": record, "labels": np.zeros((1, hybrid_model.model.num_labels), dtype=np.float32)}]
    )

    def predict_label_proba(texts: list[str]) -> np.ndarray:
        encoded = hybrid_model.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=hybrid_model.max_length,
            return_tensors="pt",
        )
        batch = {
            "input_ids": encoded["input_ids"].to(hybrid_model.device),
            "attention_mask": encoded["attention_mask"].to(hybrid_model.device),
            "x": base_batch["x"].to(hybrid_model.device),
            "adj": base_batch["adj"].to(hybrid_model.device),
            "mask": base_batch["mask"].to(hybrid_model.device),
        }
        if "cross_contract" in base_batch:
            batch["cross_contract"] = base_batch["cross_contract"].to(hybrid_model.device)
        hybrid_model.model.eval()
        with torch.no_grad():
            logits = hybrid_model.model(**batch)
            probs = torch.sigmoid(logits)[:, label_index].detach().cpu().numpy()
        return probs.astype(np.float32)

    return predict_label_proba


def save_shap_summary(summary: ShapRunSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": summary.model,
        "run_dir": summary.run_dir,
        "label": summary.label,
        "num_samples": summary.num_samples,
        "num_explained": summary.num_explained,
        "background_samples": summary.background_samples,
        "max_evals": summary.max_evals,
        "top_global_tokens": summary.top_global_tokens,
        "samples": [
            {
                **asdict(sample),
                "top_attributions": _top_attributions(sample.attributions),
            }
            for sample in summary.samples
        ],
    }
    json_path = output_dir / "shap_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_lines = [
        "# SHAP Explanation Report",
        "",
        f"- Model: `{summary.model}`",
        f"- Run: `{summary.run_dir}`",
        f"- Target label: `{summary.label}`",
        f"- Samples explained: `{summary.num_explained}` / `{summary.num_samples}`",
        "",
        "## Global influential tokens/features",
        "",
    ]
    for item in summary.top_global_tokens[:15]:
        if "token" in item:
            report_lines.append(
                f"- `{item['token']}`: mean |SHAP| = {item['mean_abs_shap']:.6f} (n={item['count']})"
            )
        else:
            report_lines.append(f"- `{item}`")

    report_lines.append("")
    for sample in summary.samples[:10]:
        report_lines.extend(
            [
                f"## {sample.function_name or 'function'} (sample {sample.sample_index})",
                "",
                f"- Contract: `{sample.contract_file}`",
                f"- P({sample.label}) = {sample.predicted_probability:.4f}",
                f"- Base value: {sample.base_value:.4f}",
                "",
                "Top attributions:",
                "",
            ]
        )
        for item in _top_attributions(sample.attributions, limit=10):
            if "token" in item:
                report_lines.append(f"- `{item['token']}`: {item['shap_value']:+.6f}")
            else:
                report_lines.append(f"- `{item['feature']}`: {item['shap_value']:+.6f}")
        report_lines.append("")

    (output_dir / "shap_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return json_path


def plot_sample_bar(sample: SampleShapExplanation, output_path: Path, limit: int = 12) -> None:
    if not _MPL_AVAILABLE:
        return
    top = _top_attributions(sample.attributions, limit=limit)
    if not top:
        return
    labels = [item.get("token") or item.get("feature") or "?" for item in top][::-1]
    values = [item["shap_value"] for item in top][::-1]
    fig, axis = plt.subplots(figsize=(8, max(3, len(labels) * 0.35)))
    axis.barh(labels, values, color=["#c44e52" if value > 0 else "#4c72b0" for value in values])
    axis.set_title(f"SHAP — {sample.label} — {sample.function_name}")
    axis.set_xlabel("SHAP value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
