#!/usr/bin/env python3
"""
Run qualitative case-study inference on a Solidity contract using a saved model run.

Typical use:
  python run_case_study_inference.py --model hybrid --run-dir experiments/hybrid_baseline/esc_hybrid_recall_push_100k --sol-file path/to/contract.sol

Notes:
  - The run directory must contain `thresholds.json`.
  - To run model inference, the run directory must also contain a saved `model/` folder
    created by training with `--save-model`.
  - If no saved model is present, the script still extracts functions and shows the
    heuristic labels, which can help with case-study preparation.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment_utils import VULN_TYPES, apply_thresholds
from models_codebert import CodeBERTMultilabelBaseline
from models_hybrid import HybridCodeBERTGNNMultilabelBaseline
from standardize_dataset import DatasetStandardizer


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_codebert_model(run_dir: Path, run_config: dict[str, Any], device: str | None):
    model_dir = run_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No saved model found in {model_dir}. Re-run training with --save-model first."
        )
    model = CodeBERTMultilabelBaseline(
        model_name=str(model_dir),
        max_length=int(run_config.get("max_length", 256)),
        eval_batch_size=int(run_config.get("eval_batch_size", 16)),
        device=device,
    )
    model._build_model(len(VULN_TYPES))
    return model


def _load_hybrid_model(run_dir: Path, device: str | None):
    model_dir = run_dir / "model"
    config_path = model_dir / "hybrid_config.json"
    state_path = model_dir / "hybrid_state.pt"
    text_encoder_dir = model_dir / "text_encoder"
    if not config_path.exists() or not state_path.exists() or not text_encoder_dir.exists():
        raise FileNotFoundError(
            f"Missing saved hybrid artifacts under {model_dir}. Re-run training with --save-model first."
        )
    cfg = _load_json(config_path)
    model = HybridCodeBERTGNNMultilabelBaseline(
        model_name=str(text_encoder_dir),
        max_length=int(cfg.get("max_length", 256)),
        max_nodes=int(cfg.get("max_nodes", 128)),
        feature_dim=int(cfg.get("feature_dim", 256)),
        graph_hidden_dim=int(cfg.get("graph_hidden_dim", 128)),
        graph_num_layers=int(cfg.get("graph_num_layers", 2)),
        fusion_dim=int(cfg.get("fusion_dim", 256)),
        attention_heads=int(cfg.get("attention_heads", 4)),
        graph_residual_scale=float(cfg.get("graph_residual_scale", 0.2)),
        dropout=float(cfg.get("dropout", 0.2)),
        eval_batch_size=8,
        device=device,
        enable_cross_contract=bool(cfg.get("enable_cross_contract", False)),
        cross_contract_use_slither=bool(cfg.get("cross_contract_use_slither", True)),
        cross_contract_residual_scale=float(cfg.get("cross_contract_residual_scale", 0.15)),
    )
    model._build_model(len(VULN_TYPES))
    state_dict = torch.load(state_path, map_location=model.device)
    assert model.model is not None
    model.model.load_state_dict(state_dict)
    return model


def _extract_records(sol_file: Path, fallback_only: bool) -> list[dict[str, Any]]:
    standardizer = DatasetStandardizer(output_dir="case_study_tmp", fallback_only=fallback_only)
    functions = standardizer.process_file(str(sol_file))
    return [
        {
            "contract_file": fn.contract_file,
            "contract_name": fn.contract_name,
            "function_name": fn.function_name,
            "function_signature": fn.function_signature,
            "function_code": fn.function_code,
            "start_line": fn.start_line,
            "end_line": fn.end_line,
            "visibility": fn.visibility,
            "state_mutability": fn.state_mutability,
            "vulnerabilities": fn.vulnerabilities,
            "swc_ids": fn.swc_ids,
            "labels": fn.labels,
            "metadata": fn.metadata,
        }
        for fn in functions
    ]


def _predict(model_name: str, model, records: list[dict[str, Any]], thresholds: dict[str, float]):
    if model_name == "codebert":
        texts = [record["function_code"] for record in records]
        y_prob = model.predict_proba(texts)
    elif model_name == "hybrid":
        y_prob = model.predict_proba(records)
    else:
        raise ValueError(f"Unsupported model for case study inference: {model_name}")
    y_pred = apply_thresholds(y_prob, thresholds, label_order=VULN_TYPES)
    return y_prob, y_pred


def _display_signature(record: dict[str, Any], index: int) -> str:
    signature = record.get("function_signature")
    if isinstance(signature, str):
        stripped = signature.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            try:
                signature = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                signature = stripped
    if isinstance(signature, str) and signature.strip():
        return signature
    if isinstance(signature, (list, tuple)) and signature:
        name = str(signature[0]) if len(signature) > 0 else ""
        params = ", ".join(str(item) for item in signature[1]) if len(signature) > 1 and signature[1] else ""
        returns = ", ".join(str(item) for item in signature[2]) if len(signature) > 2 and signature[2] else ""
        rendered = f"{name}({params})"
        if returns:
            rendered += f" returns ({returns})"
        return rendered
    fallback_name = str(record.get("function_name") or "").strip()
    return fallback_name or f"function_{index + 1}"


def _build_report(
    *,
    sol_file: Path,
    model_name: str,
    run_dir: Path,
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    y_prob: np.ndarray | None,
    y_pred: np.ndarray | None,
) -> str:
    lines = [
        f"# Case Study Inference Report",
        "",
        f"- Contract file: `{sol_file}`",
        f"- Model: `{model_name}`",
        f"- Run directory: `{run_dir}`",
        f"- Functions extracted: `{len(records)}`",
        "",
    ]
    if y_prob is None or y_pred is None:
        lines.extend(
            [
                "> No saved model artifacts were available, so only heuristic labels are shown.",
                "",
            ]
        )

    for idx, record in enumerate(records):
        lines.extend(
            [
                f"## {_display_signature(record, idx)}",
                "",
                f"- Contract: `{record['contract_name']}`",
                f"- Lines: `{record['start_line']}-{record['end_line']}`",
                f"- Heuristic labels: `{', '.join(record['vulnerabilities']) if record['vulnerabilities'] else 'none'}`",
                f"- Heuristic SWC IDs: `{', '.join(record['swc_ids']) if record['swc_ids'] else 'none'}`",
            ]
        )

        if y_prob is not None and y_pred is not None:
            predicted_labels = [VULN_TYPES[i] for i, value in enumerate(y_pred[idx]) if value]
            lines.append(
                f"- Model predicted labels: `{', '.join(predicted_labels) if predicted_labels else 'none'}`"
            )
            lines.append("- Model probabilities:")
            for label_idx, label in enumerate(VULN_TYPES):
                lines.append(
                    f"  - `{label}`: prob={float(y_prob[idx, label_idx]):.4f}, threshold={thresholds[label]:.2f}"
                )

        snippet = record["function_code"].strip()
        if snippet:
            lines.extend(
                [
                    "",
                    "```solidity",
                    snippet,
                    "```",
                ]
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run qualitative contract inference for case studies.")
    parser.add_argument("--model", choices=["codebert", "hybrid"], required=True)
    parser.add_argument("--run-dir", required=True, help="Run directory containing thresholds.json and optional saved model/")
    parser.add_argument("--sol-file", required=True, help="Path to the Solidity contract file to inspect.")
    parser.add_argument("--output", help="Optional markdown report output path.")
    parser.add_argument("--device", help="Optional torch device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Use regex fallback extraction only when parsing the Solidity file.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    sol_file = Path(args.sol_file)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not sol_file.exists():
        raise FileNotFoundError(f"Solidity file not found: {sol_file}")

    thresholds_path = run_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing thresholds file: {thresholds_path}")
    thresholds = _load_json(thresholds_path)
    run_config_path = run_dir / "run_config.json"
    run_config = _load_json(run_config_path) if run_config_path.exists() else {}

    records = _extract_records(sol_file, fallback_only=args.fallback_only)
    if not records:
        raise RuntimeError(f"No functions extracted from {sol_file}")

    y_prob = None
    y_pred = None
    try:
        if args.model == "codebert":
            model = _load_codebert_model(run_dir, run_config, args.device)
        else:
            model = _load_hybrid_model(run_dir, args.device)
        y_prob, y_pred = _predict(args.model, model, records, thresholds)
    except FileNotFoundError as exc:
        print(f"[warn] {exc}")
        print("[warn] Continuing with heuristic labels only.")

    report = _build_report(
        sol_file=sol_file,
        model_name=args.model,
        run_dir=run_dir,
        records=records,
        thresholds=thresholds,
        y_prob=y_prob,
        y_pred=y_pred,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Saved report to {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
