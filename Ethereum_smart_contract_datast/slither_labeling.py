#!/usr/bin/env python3
"""
Slither detector-based function labeling shared by standardize_dataset.py and
models_slither_baseline.py.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from report_vulnerability_counts import VULN_TYPES

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


def finding_vuln_labels(check: str, description: str = "") -> list[str]:
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
def all_slither_detector_classes() -> tuple[type, ...]:
    from slither.detectors import all_detectors
    from slither.detectors.abstract_detector import AbstractDetector

    detectors = [
        getattr(all_detectors, name)
        for name in dir(all_detectors)
        if inspect.isclass(getattr(all_detectors, name))
        and issubclass(getattr(all_detectors, name), AbstractDetector)
    ]
    return tuple(detectors)


def element_line_numbers(element) -> set[int]:
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


def run_slither_contract_vuln_lines(
    contract_file: str,
    *,
    detector_filter: list[str] | None = None,
) -> dict[str, set[int]]:
    """
    Run Slither detectors on a .sol file.

    Returns mapping: vulnerability_label -> set of source line numbers.
    """
    from slither import Slither

    slither = Slither(contract_file)
    detector_classes = all_slither_detector_classes()
    if detector_filter:
        allowed = {name.strip().lower() for name in detector_filter}
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
        labels = finding_vuln_labels(check, description)
        line_numbers: set[int] = set()
        for element in result.get("elements", []) or []:
            line_numbers.update(element_line_numbers(element))
        if not line_numbers or not labels:
            continue
        for label in labels:
            if label in VULN_TYPES:
                vuln_lines[label].update(line_numbers)
    return dict(vuln_lines)


def label_function_from_vuln_lines(
    start_line: int,
    end_line: int,
    vuln_lines: dict[str, set[int]],
) -> list[str]:
    """Assign vulnerability type names whose Slither findings overlap the function span."""
    if end_line <= 0:
        end_line = start_line
    vulnerabilities: list[str] = []
    for label in VULN_TYPES:
        lines = vuln_lines.get(label)
        if not lines:
            continue
        if any(start_line <= line <= end_line for line in lines):
            vulnerabilities.append(label)
    return vulnerabilities


class SlitherFunctionLabeler:
    """Contract-level Slither analysis with per-function label assignment."""

    def __init__(
        self,
        *,
        detector_filter: list[str] | None = None,
        fail_on_compile_error: bool = False,
    ):
        self.detector_filter = detector_filter
        self.fail_on_compile_error = fail_on_compile_error
        self._cache: dict[str, dict[str, set[int]]] = {}
        self.stats = {
            "contracts_analyzed": 0,
            "contracts_failed": 0,
            "functions_labeled": 0,
            "functions_with_any_vuln": 0,
        }

    def contract_vulnerability_lines(self, contract_file: str) -> dict[str, set[int]]:
        resolved = str(Path(contract_file).resolve())
        if resolved in self._cache:
            return self._cache[resolved]

        try:
            vuln_lines = run_slither_contract_vuln_lines(
                resolved,
                detector_filter=self.detector_filter,
            )
            self._cache[resolved] = vuln_lines
            self.stats["contracts_analyzed"] += 1
            return vuln_lines
        except Exception as exc:
            if self.fail_on_compile_error:
                raise
            print(f"[slither-label] Skipping {contract_file}: {exc}")
            self._cache[resolved] = {}
            self.stats["contracts_failed"] += 1
            return {}

    def label_function(
        self,
        *,
        start_line: int,
        end_line: int,
        contract_file: str,
    ) -> tuple[list[str], dict[str, bool]]:
        vuln_lines = self.contract_vulnerability_lines(contract_file)
        vulnerabilities = label_function_from_vuln_lines(start_line, end_line, vuln_lines)
        self.stats["functions_labeled"] += 1
        if vulnerabilities:
            self.stats["functions_with_any_vuln"] += 1
        detailed = {label: True for label in vulnerabilities}
        return vulnerabilities, detailed

    def format_stats(self) -> str:
        return (
            f"contracts_analyzed={self.stats['contracts_analyzed']} "
            f"contracts_failed={self.stats['contracts_failed']} "
            f"functions_labeled={self.stats['functions_labeled']} "
            f"functions_with_any_vuln={self.stats['functions_with_any_vuln']}"
        )
