#!/usr/bin/env python3
"""
Generate SHAP explanations for a trained experiment run.

Examples:
  py run_shap_explain.py --model codebert --run-dir experiments/codebert_baseline/esc_codebert_tuned_100k \\
    --split-dir experiment_splits/esc_primary --split val --label Reentrancy --max-samples 10

  py run_shap_explain.py --model random_forest --run-dir experiments/random_forest_baseline/esc_rf_200k \\
    --split-dir experiment_splits/esc_primary --split val --label Reentrancy --max-samples 10

  py run_shap_explain.py --model hybrid --run-dir experiments/hybrid_baseline/esc_hybrid_recall_push_100k \\
    --sol-file 0.sol --label Reentrancy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment_utils import VULN_TYPES, load_named_split
from models_codebert import CodeBERTMultilabelBaseline
from models_hybrid import HybridCodeBERTGNNMultilabelBaseline
from models_tabular import RandomForestMultilabelBaseline, TabularMultilabelBaseline
from run_case_study_inference import _extract_records, _load_codebert_model, _load_hybrid_model
from shap_explain import (
    _global_token_importance,
    build_hybrid_text_predict_fn,
    explain_tabular_label,
    explain_text_model_label,
    plot_sample_bar,
    require_shap,
    save_shap_summary,
    ShapRunSummary,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_label(label: str | None, label_index: int | None) -> tuple[str, int]:
    if label is not None:
        if label not in VULN_TYPES:
            raise ValueError(f"Unknown label {label!r}. Choose from: {VULN_TYPES}")
        return label, VULN_TYPES.index(label)
    if label_index is not None:
        if label_index < 0 or label_index >= len(VULN_TYPES):
            raise ValueError(f"label_index must be in [0, {len(VULN_TYPES) - 1}]")
        return VULN_TYPES[label_index], label_index
    return "Reentrancy", VULN_TYPES.index("Reentrancy")


def _load_tabular_model(run_config: dict, train_split) -> TabularMultilabelBaseline:
    model = TabularMultilabelBaseline(
        max_features=int(run_config.get("max_features", 50000)),
        min_df=int(run_config.get("min_df", 2)),
        max_df=float(run_config.get("max_df", 0.95)),
        c_value=float(run_config.get("c_value", 4.0)),
        max_iter=int(run_config.get("max_iter", 1000)),
    )
    print(f"[shap] Fitting tabular model on {len(train_split.texts)} train samples for SHAP background...")
    model.fit(train_split.texts, train_split.labels)
    return model


def _load_random_forest_model(run_config: dict, train_split) -> RandomForestMultilabelBaseline:
    model = RandomForestMultilabelBaseline(
        max_features=int(run_config.get("max_features", 50000)),
        min_df=int(run_config.get("min_df", 2)),
        max_df=float(run_config.get("max_df", 0.95)),
        n_estimators=int(run_config.get("rf_n_estimators", 200)),
        max_depth=run_config.get("rf_max_depth"),
        min_samples_leaf=int(run_config.get("rf_min_samples_leaf", 1)),
        random_state=int(run_config.get("seed", 42)),
    )
    print(f"[shap] Fitting Random Forest model on {len(train_split.texts)} train samples for SHAP background...")
    model.fit(train_split.texts, train_split.labels)
    return model


def _select_positive_samples(
    texts: list[str],
    records: list[dict],
    probs: np.ndarray,
    label_index: int,
    *,
    max_samples: int,
    min_probability: float,
) -> tuple[list[str], list[dict], np.ndarray]:
    indices = [idx for idx, prob in enumerate(probs[:, label_index]) if prob >= min_probability]
    if not indices:
        indices = list(np.argsort(probs[:, label_index])[-max_samples:][::-1])
    indices = indices[:max_samples]
    selected_texts = [texts[idx] for idx in indices]
    selected_records = [records[idx] for idx in indices]
    selected_probs = probs[indices]
    return selected_texts, selected_records, selected_probs


def main():
    parser = argparse.ArgumentParser(description="SHAP explanations for experiment runs.")
    parser.add_argument("--model", choices=["tabular", "random_forest", "codebert", "hybrid"], required=True)
    parser.add_argument("--run-dir", required=True, help="Experiment run directory with run_config.json")
    parser.add_argument("--split-dir", help="Split directory with train/val/test JSON")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sol-file", help="Explain functions extracted from one Solidity file instead of a split.")
    parser.add_argument("--label", help=f"Target vulnerability label (default: Reentrancy). One of {VULN_TYPES}")
    parser.add_argument("--label-index", type=int, help="Target label index [0-6].")
    parser.add_argument("--max-samples", type=int, default=10, help="Max functions to explain.")
    parser.add_argument("--background-samples", type=int, default=25, help="Background samples for SHAP.")
    parser.add_argument("--background-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--min-probability", type=float, default=0.5, help="Prefer samples above this predicted probability.")
    parser.add_argument("--max-evals", type=int, default=100, help="SHAP max_evals (lower = faster).")
    parser.add_argument("--output-dir", help="Output directory (default: <run-dir>/shap).")
    parser.add_argument("--device", help="Torch device for codebert/hybrid, e.g. cpu or cuda")
    parser.add_argument("--contract-root", help="Project root for resolving contract_file paths.")
    parser.add_argument("--contracts-dir", help="Path to contract_dataset_ethereum if not under cwd.")
    parser.add_argument("--fallback-only", action="store_true", help="Regex-only extraction for --sol-file.")
    args = parser.parse_args()

    require_shap()
    label_name, label_index = _resolve_label(args.label, args.label_index)
    run_dir = Path(args.run_dir)
    run_config = _load_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "shap" / label_name.replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sol_file:
        records = _extract_records(Path(args.sol_file), fallback_only=args.fallback_only)
        texts = [record["function_code"] for record in records]
        background_texts = texts[:1] if texts else [""]
    else:
        if not args.split_dir:
            raise ValueError("Provide --split-dir or --sol-file.")
        project_root = Path(args.contract_root).resolve() if args.contract_root else Path.cwd().resolve()
        contracts_dir = Path(args.contracts_dir).resolve() if args.contracts_dir else None
        eval_split = load_named_split(
            args.split,
            Path(args.split_dir) / f"{args.split}.json",
            max_samples=args.max_samples * 4,
            project_root=project_root,
            contracts_dir=contracts_dir,
        )
        background_split = load_named_split(
            args.background_split,
            Path(args.split_dir) / f"{args.background_split}.json",
            max_samples=args.background_samples,
            seed=17,
            project_root=project_root,
            contracts_dir=contracts_dir,
        )
        records = eval_split.records
        texts = eval_split.texts
        background_texts = background_split.texts

    if not texts:
        raise RuntimeError("No samples available for SHAP.")

    print(f"[shap] Explaining label={label_name} for model={args.model} on {len(texts)} candidate functions...")

    if args.model in {"tabular", "random_forest"}:
        if args.sol_file:
            raise ValueError(f"{args.model} SHAP requires --split-dir to refit TF-IDF on train split.")
        train_split = load_named_split(
            "train",
            Path(args.split_dir) / "train.json",
            max_samples=max(5000, args.background_samples * 4),
            project_root=Path(args.contract_root).resolve() if args.contract_root else Path.cwd(),
            contracts_dir=Path(args.contracts_dir).resolve() if args.contracts_dir else None,
        )
        if args.model == "tabular":
            model = _load_tabular_model(run_config, train_split)
        else:
            model = _load_random_forest_model(run_config, train_split)
        probs = model.predict_proba(texts)
        selected_texts, selected_records, _ = _select_positive_samples(
            texts,
            records,
            probs,
            label_index,
            max_samples=args.max_samples,
            min_probability=args.min_probability,
        )
        for idx, record in enumerate(selected_records):
            record.setdefault("contract_file", record.get("contract_file", ""))
            record.setdefault("function_name", record.get("function_name", f"fn_{idx}"))
        explanations = explain_tabular_label(
            model,
            texts=selected_texts,
            label=label_name,
            label_index=label_index,
            background_texts=background_texts[: args.background_samples],
            max_evals=args.max_evals,
        )
    elif args.model == "codebert":
        model = _load_codebert_model(run_dir, run_config, args.device)
        probs = model.predict_proba(texts)
        selected_texts, selected_records, _ = _select_positive_samples(
            texts,
            records,
            probs,
            label_index,
            max_samples=args.max_samples,
            min_probability=args.min_probability,
        )

        def predict_fn(masked_texts: list[str]) -> np.ndarray:
            return model.predict_proba(masked_texts)[:, label_index]

        explanations = explain_text_model_label(
            predict_label_proba=predict_fn,
            tokenizer=model.tokenizer,
            texts=selected_texts,
            records=selected_records,
            label=label_name,
            label_index=label_index,
            background_texts=background_texts[: args.background_samples],
            max_evals=args.max_evals,
        )
    else:
        model = _load_hybrid_model(run_dir, args.device)
        probs = model.predict_proba(records)
        selected_texts, selected_records, _ = _select_positive_samples(
            texts,
            records,
            probs,
            label_index,
            max_samples=args.max_samples,
            min_probability=args.min_probability,
        )
        explanations = []
        for idx, (text, record) in enumerate(zip(selected_texts, selected_records)):
            print(f"[shap] Hybrid text-branch SHAP {idx + 1}/{len(selected_records)}: {record.get('function_name', '')}")
            predict_fn = build_hybrid_text_predict_fn(model, record, label_index)
            sample_explanations = explain_text_model_label(
                predict_label_proba=predict_fn,
                tokenizer=model.tokenizer,
                texts=[text],
                records=[record],
                label=label_name,
                label_index=label_index,
                background_texts=background_texts[: max(3, min(5, args.background_samples))],
                max_evals=args.max_evals,
            )
            for sample in sample_explanations:
                sample.sample_index = idx
            explanations.extend(sample_explanations)

    summary = ShapRunSummary(
        model=args.model,
        run_dir=str(run_dir.resolve()),
        label=label_name,
        num_samples=len(texts),
        num_explained=len(explanations),
        background_samples=min(args.background_samples, len(background_texts)),
        max_evals=args.max_evals,
        samples=explanations,
    )
    summary.top_global_tokens = _global_token_importance(explanations)

    json_path = save_shap_summary(summary, output_dir)
    for sample in explanations[:5]:
        plot_sample_bar(sample, output_dir / f"sample_{sample.sample_index:03d}_{label_name.replace(' ', '_')}.png")

    print(f"[shap] Saved {len(explanations)} explanations to {output_dir}")
    print(f"[shap] Summary: {json_path}")
    print(f"[shap] Report: {output_dir / 'shap_report.md'}")


if __name__ == "__main__":
    main()
