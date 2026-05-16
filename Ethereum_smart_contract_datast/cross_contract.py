#!/usr/bin/env python3
"""
Cross-contract interaction context for function-level records.

Builds a lightweight call graph within each contract directory (e.g. all .sol
files under contract_dataset_ethereum/contract1/) using Slither when available,
with regex fallback for high-level external calls.

The graph is used to attach structural cross-contract features to each function:
- how many other contracts it calls (out-degree to other files)
- how many other contracts call into it (in-degree from other files)
- total internal call fan-out / fan-in inside the same project folder

These features are fused in the hybrid model when --hybrid-enable-cross-contract
is enabled. They do not require labels from neighboring contracts, which avoids
label leakage from the test split into training features.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CROSS_CONTRACT_VECTOR_DIM = 4

_FUNCTION_KEY = tuple[str, str, str, int, int]


def function_key(record: dict) -> _FUNCTION_KEY:
    return (
        str(record.get("contract_file", "")),
        str(record.get("contract_name", "")),
        str(record.get("function_name", "")),
        int(record.get("start_line", 0) or 0),
        int(record.get("end_line", 0) or 0),
    )


@dataclass
class CrossContractFeatures:
    in_degree: int = 0
    out_degree: int = 0
    unique_callee_contracts: int = 0
    unique_caller_contracts: int = 0
    callee_contract_files: list[str] = field(default_factory=list)
    caller_contract_files: list[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                float(np.log1p(self.in_degree)),
                float(np.log1p(self.out_degree)),
                float(np.log1p(self.unique_callee_contracts)),
                float(np.log1p(self.unique_caller_contracts)),
            ],
            dtype=np.float32,
        )

    def to_metadata(self) -> dict:
        return {
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "unique_callee_contracts": self.unique_callee_contracts,
            "unique_caller_contracts": self.unique_caller_contracts,
            "callee_contract_files": self.callee_contract_files[:20],
            "caller_contract_files": self.caller_contract_files[:20],
        }


@dataclass
class CrossContractIndex:
    """Call graph index scoped by parent directory of .sol files."""

    features_by_key: dict[_FUNCTION_KEY, CrossContractFeatures] = field(default_factory=dict)
    project_dirs: list[str] = field(default_factory=list)

    def get(self, record: dict) -> CrossContractFeatures:
        return self.features_by_key.get(
            function_key(record),
            CrossContractFeatures(),
        )

    def enrich_record(self, record: dict) -> dict:
        enriched = dict(record)
        metadata = dict(enriched.get("metadata") or {})
        features = self.get(record)
        metadata["cross_contract"] = features.to_metadata()
        enriched["metadata"] = metadata
        enriched["cross_contract_vector"] = features.to_vector().tolist()
        return enriched

    def enrich_records(self, records: list[dict]) -> list[dict]:
        return [self.enrich_record(record) for record in records]


def build_cross_contract_index(
    records: list[dict],
    *,
    use_slither: bool = True,
) -> CrossContractIndex:
    """
    Build cross-contract edges for all records, grouped by the parent folder of
    each contract_file (typical ESC layout: contract_dataset_ethereum/contractN/).
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        contract_file = str(record.get("contract_file", ""))
        if not contract_file:
            continue
        grouped[str(Path(contract_file).resolve().parent)].append(record)

    key_to_features: dict[_FUNCTION_KEY, CrossContractFeatures] = {}
    project_dirs = sorted(grouped.keys())

    for project_dir, project_records in grouped.items():
        key_map = {function_key(record): record for record in project_records}
        edges = _build_project_call_edges(project_dir, project_records, use_slither=use_slither)

        out_neighbors: dict[_FUNCTION_KEY, set[_FUNCTION_KEY]] = defaultdict(set)
        in_neighbors: dict[_FUNCTION_KEY, set[_FUNCTION_KEY]] = defaultdict(set)
        out_contracts: dict[_FUNCTION_KEY, set[str]] = defaultdict(set)
        in_contracts: dict[_FUNCTION_KEY, set[str]] = defaultdict(set)

        for caller_key, callee_key in edges:
            if caller_key == callee_key:
                continue
            caller_file = caller_key[0]
            callee_file = callee_key[0]
            if Path(caller_file).resolve() == Path(callee_file).resolve():
                continue
            out_neighbors[caller_key].add(callee_key)
            in_neighbors[callee_key].add(caller_key)
            out_contracts[caller_key].add(callee_file)
            in_contracts[callee_key].add(caller_file)

        for record in project_records:
            key = function_key(record)
            callee_files = sorted(out_contracts.get(key, set()))
            caller_files = sorted(in_contracts.get(key, set()))
            key_to_features[key] = CrossContractFeatures(
                in_degree=len(in_neighbors.get(key, set())),
                out_degree=len(out_neighbors.get(key, set())),
                unique_callee_contracts=len(callee_files),
                unique_caller_contracts=len(caller_files),
                callee_contract_files=callee_files,
                caller_contract_files=caller_files,
            )

    return CrossContractIndex(features_by_key=key_to_features, project_dirs=project_dirs)


def _build_project_call_edges(
    project_dir: str,
    records: list[dict],
    *,
    use_slither: bool,
) -> list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]]:
    edges: list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]] = []
    if use_slither:
        try:
            edges.extend(_build_edges_with_slither(project_dir, records))
        except Exception:
            edges = []
    if not edges:
        edges.extend(_build_edges_with_regex(records))
    return edges


def _build_edges_with_slither(
    project_dir: str,
    records: list[dict],
) -> list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]]:
    from slither import Slither

    sol_files = sorted(Path(project_dir).glob("*.sol"))
    if not sol_files:
        return []

    # Slither accepts multiple inputs for one compilation unit when possible.
    try:
        slither = Slither([str(path) for path in sol_files])
    except Exception:
        # Fall back to per-file analysis if joint compilation fails.
        edges: list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]] = []
        for sol_file in sol_files:
            file_records = [r for r in records if Path(r["contract_file"]).resolve() == sol_file.resolve()]
            if not file_records:
                continue
            try:
                edges.extend(_build_edges_for_slither_instance(Slither(str(sol_file)), file_records))
            except Exception:
                continue
        return edges

    return _build_edges_for_slither_instance(slither, records)


def _build_edges_for_slither_instance(
    slither,
    records: list[dict],
) -> list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]]:
    index_by_name: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        index_by_name[(str(record.get("contract_name", "")), str(record.get("function_name", "")))].append(record)

    edges: list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]] = []
    for contract in getattr(slither, "contracts", []) or []:
        contract_name = getattr(contract, "name", "")
        for function in getattr(contract, "functions", []) or []:
            caller_candidates = _match_records_for_slither_function(function, contract_name, index_by_name)
            if not caller_candidates:
                continue
            caller_key = function_key(caller_candidates[0])

            call_iterables = []
            for attr in ("internal_calls", "high_level_calls", "low_level_calls"):
                call_iterables.extend(getattr(function, attr, []) or [])

            for call in call_iterables:
                callee_contract = getattr(call, "contract", None)
                callee_function = getattr(call, "function", None)
                if callee_contract is None or callee_function is None:
                    continue
                callee_name = getattr(callee_contract, "name", "")
                callee_fn_name = getattr(callee_function, "name", "")
                callee_records = index_by_name.get((callee_name, callee_fn_name), [])
                if not callee_records:
                    continue
                callee_key = function_key(callee_records[0])
                edges.append((caller_key, callee_key))

    return edges


def _match_records_for_slither_function(
    function,
    contract_name: str,
    index_by_name: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    function_name = getattr(function, "name", "")
    candidates = list(index_by_name.get((contract_name, function_name), []))
    if candidates:
        return candidates

    # Constructor / fallback names
    kind = getattr(function, "function_type", None)
    if kind is not None:
        kind_name = str(getattr(kind, "name", kind)).lower()
        if kind_name == "constructor":
            candidates = list(index_by_name.get((contract_name, "constructor"), []))
    return candidates


def _build_edges_with_regex(records: list[dict]) -> list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]]:
    contract_names = {str(record.get("contract_name", "")) for record in records}
    call_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9_]*)\s*\(\s*[^)]*\)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)",
        re.MULTILINE,
    )
    edges: list[tuple[_FUNCTION_KEY, _FUNCTION_KEY]] = []
    records_by_contract: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_contract[str(record.get("contract_name", ""))].append(record)

    for record in records:
        caller_key = function_key(record)
        code = str(record.get("function_code", ""))
        for match in call_pattern.finditer(code):
            callee_contract_name = match.group(1)
            callee_function_name = match.group(2)
            if callee_contract_name not in contract_names:
                continue
            callee_records = records_by_contract.get(callee_contract_name, [])
            for callee_record in callee_records:
                if callee_record.get("function_name") == callee_function_name:
                    edges.append((caller_key, function_key(callee_record)))
                    break
    return edges
