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

import inspect
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiment_utils import VULN_TYPES

# Slither check names (partial) -> project vulnerability label
SLITHER_CHECK_TO_VULN: dict[str, str] = {
    "reentrancy-eth": "Reentrancy",
    "reentrancy-no-eth": "Reentrancy",
    "reentrancy-benign": "Reentrancy",
    "reentrancy-events": "Reentrancy",
    "timestamp": "Timestamp Dependency",
    "weak-prng": "Timestamp Dependency",
    "incorrect-equality": "Timestamp Dependency",
    "integer-overflow": "Integer Overflow/Underflow",
    "incorrect-exp": "Integer Overflow/Underflow",
    "divide-before-multiply": "Integer Overflow/Underflow",
    "delegatecall": "Dangerous Delegatecall",
    "controlled-delegatecall": "Dangerous Delegatecall",
    "delegatecall-loop": "Dangerous Delegatecall",
    "tx-origin": "Transaction-Ordering Dependence",
    "suicidal": "Transaction-Ordering Dependence",
    "arbitrary-send-eth": "Transaction-Ordering Dependence",
    "arbitrary-send-erc20": "Transaction-Ordering Dependence",
    "uninitialized-storage": "Uninitialized Storage Pointer",
    "uninitialized-state": "Uninitialized Storage Pointer",
    "unused-state": "Uninitialized Storage Pointer",
    "unchecked-lowlevel": "Unchecked External Calls",
    "unchecked-send": "Unchecked External Calls",
    "unchecked-transfer": "Unchecked External Calls",
    "return-bomb": "Unchecked External Calls",
    "low-level-calls": "Unchecked External Calls",
}


def _normalize_check_name(check: str) -> str:
    return check.strip().lower().replace(" ", "-")


def _finding_vuln_labels(check: str, description: str = "") -> list[str]:
    normalized = _normalize_check_name(check)
    if normalized in SLITHER_CHECK_TO_VULN:
        return [SLITHER_CHECK_TO_VULN[normalized]]

    description_lower = description.lower()
    fallback: list[str] = []
    if "reentranc" in description_lower:
        fallback.append("Reentrancy")
    if "timestamp" in description_lower or "block.timestamp" in description_lower:
        fallback.append("Timestamp Dependency")
    if "overflow" in description_lower or "underflow" in description_lower:
        fallback.append("Integer Overflow/Underflow")
    if "delegatecall" in description_lower:
        fallback.append("Dangerous Delegatecall")
    if "tx.origin" in description_lower or "transaction-order" in description_lower:
        fallback.append("Transaction-Ordering Dependence")
    if "uninitialized" in description_lower and "storage" in description_lower:
        fallback.append("Uninitialized Storage Pointer")
    if "unchecked" in description_lower or "return value" in description_lower:
        fallback.append("Unchecked External Calls")
    return fallback


@lru_cache(maxsize=1)
def _all_slither_detector_classes() -> tuple[type, ...]:
    from slither.detectors import all_detectors
    from slither.detectors.abstract_detector import AbstractDetector

    detectors = [
        getattr(all_detectors, name)
        for name in dir(all_detectors)
        if inspect.isclass(getattr(all_detectors, name))
        and issubclass(getattr(all_detectors, name), AbstractDetector)
    ]
    return tuple(detectors)


def _element_line_numbers(element) -> set[int]:
    if isinstance(element, dict):
        source_mapping = element.get("source_mapping") or {}
        if isinstance(source_mapping, dict):
            lines = source_mapping.get("lines") or []
        else:
            lines = getattr(source_mapping, "lines", None) or []
    else:
        source_mapping = getattr(element, "source_mapping", None)
        lines = getattr(source_mapping, "lines", None) or [] if source_mapping else []

    line_numbers: set[int] = set()
    for line in lines:
        try:
            line_numbers.add(int(line))
        except (TypeError, ValueError):
            continue
    return line_numbers


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
        self.detectors = detectors
        self.fail_on_compile_error = fail_on_compile_error
        self.default_probability = float(default_probability)
        self._contract_cache: dict[str, dict[str, set[int]]] = {}
        self.label_to_index = {label: idx for idx, label in enumerate(VULN_TYPES)}

    def fit(self, train_records: list[dict], train_labels: np.ndarray | None = None):
        return self

    def _run_slither_on_contract(self, contract_file: str) -> dict[str, set[int]]:
        """
        Returns mapping vulnerability_label -> set of source line numbers with hits.
        """
        from slither import Slither

        slither = Slither(contract_file)
        detector_classes = _all_slither_detector_classes()
        if self.detectors:
            allowed = {name.strip().lower() for name in self.detectors}
            detector_classes = tuple(
                detector
                for detector in detector_classes
                if getattr(detector, "ARGUMENT", "").lower() in allowed
            )
        for detector_cls in detector_classes:
            slither.register_detector(detector_cls)

        raw_results = slither.run_detectors()
        findings = [
            item for sublist in raw_results if sublist for item in sublist if isinstance(item, dict)
        ]

        vuln_lines: dict[str, set[int]] = defaultdict(set)
        for result in findings:
            check = str(result.get("check", "") or "")
            description = str(result.get("description", "") or "")
            labels = _finding_vuln_labels(check, description)
            line_numbers: set[int] = set()
            for element in result.get("elements", []) or []:
                line_numbers.update(_element_line_numbers(element))
            if not line_numbers or not labels:
                continue
            for label in labels:
                if label in self.label_to_index:
                    vuln_lines[label].update(line_numbers)
        return dict(vuln_lines)

    def _contract_findings(self, contract_file: str) -> dict[str, set[int]]:
        resolved = str(Path(contract_file).resolve())
        if resolved not in self._contract_cache:
            try:
                self._contract_cache[resolved] = self._run_slither_on_contract(resolved)
            except Exception as exc:
                if self.fail_on_compile_error:
                    raise
                print(f"[slither] Skipping {contract_file}: {exc}")
                self._contract_cache[resolved] = {}
        return self._contract_cache[resolved]

    def _function_hit(
        self,
        record: dict,
        vuln_lines: dict[str, set[int]],
    ) -> np.ndarray:
        start_line = int(record.get("start_line", 0) or 0)
        end_line = int(record.get("end_line", 0) or 0)
        if end_line <= 0:
            end_line = start_line
        probs = np.zeros(len(VULN_TYPES), dtype=np.float32)
        for label, lines in vuln_lines.items():
            idx = self.label_to_index.get(label)
            if idx is None:
                continue
            if any(start_line <= line <= end_line for line in lines):
                probs[idx] = self.default_probability
        return probs

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        if not records:
            return np.zeros((0, len(VULN_TYPES)), dtype=np.float32)
        outputs = np.zeros((len(records), len(VULN_TYPES)), dtype=np.float32)
        for idx, record in enumerate(records):
            contract_file = str(record.get("contract_file", ""))
            if not contract_file or not Path(contract_file).exists():
                continue
            vuln_lines = self._contract_findings(contract_file)
            outputs[idx] = self._function_hit(record, vuln_lines)
        return outputs
