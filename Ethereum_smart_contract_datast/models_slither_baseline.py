#!/usr/bin/env python3
"""
Slither static-analysis detector baseline for function-level multilabel evaluation.

Runs Slither on each unique contract file, maps detector hits to the project's
seven vulnerability types, and assigns findings to functions by source line range.

This is a traditional-tool baseline (no training). Use it in RQ comparisons against
CodeBERT, GNN, and Hybrid models via:

  python train_experiment.py --model slither --split-dir experiment_splits/esc_primary
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_utils import VULN_TYPES, resolve_contract_path
from slither_labeling import SlitherFunctionLabeler, label_function_from_vuln_lines


class SlitherDetectorMultilabelBaseline:
    """
    Function-level multilabel predictor using Slither detectors only.
    """

    def __init__(
        self,
        *,
        detectors: list[str] | None = None,
        fail_on_compile_error: bool = False,
        default_probability: float = 1.0,
    ):
        self.default_probability = float(default_probability)
        self._labeler = SlitherFunctionLabeler(
            detector_filter=detectors,
            fail_on_compile_error=fail_on_compile_error,
        )
        self._predict_stats = {
            "records": 0,
            "missing_contract": 0,
        }

    def fit(self, train_records: list[dict], train_labels: np.ndarray | None = None):
        return self

    def _resolve_contract_file(self, contract_file: str) -> str | None:
        path = resolve_contract_path(contract_file, project_root=Path.cwd())
        if path is None:
            return None
        return str(path.resolve())

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        if not records:
            return np.zeros((0, len(VULN_TYPES)), dtype=np.float32)
        outputs = np.zeros((len(records), len(VULN_TYPES)), dtype=np.float32)
        self._predict_stats = {"records": len(records), "missing_contract": 0}

        for idx, record in enumerate(records):
            contract_file = str(record.get("contract_file", ""))
            resolved = self._resolve_contract_file(contract_file) if contract_file else None
            if resolved is None:
                self._predict_stats["missing_contract"] += 1
                continue

            vuln_lines = self._labeler.contract_vulnerability_lines(resolved)
            vulnerabilities = label_function_from_vuln_lines(
                int(record.get("start_line", 0) or 0),
                int(record.get("end_line", 0) or 0),
                vuln_lines,
            )
            for label in vulnerabilities:
                if label in VULN_TYPES:
                    label_idx = VULN_TYPES.index(label)
                    outputs[idx, label_idx] = self.default_probability

        print(
            "[slither] predict_proba stats: "
            f"records={self._predict_stats['records']} "
            f"missing_contract={self._predict_stats['missing_contract']} "
            f"{self._labeler.format_stats()}"
        )
        return outputs
