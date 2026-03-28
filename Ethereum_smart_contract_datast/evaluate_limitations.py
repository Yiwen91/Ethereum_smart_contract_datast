#!/usr/bin/env python3
"""
Empirical evaluation helpers for analyzing labeling/tool limitations.

Examples:
  py evaluate_limitations.py summary --from-json standardized_dataset/standardized_dataset.json
  py evaluate_limitations.py overflow-08 --from-json standardized_dataset/standardized_dataset.json
  py evaluate_limitations.py reentrancy-guards --from-json standardized_dataset/standardized_dataset.json
  py evaluate_limitations.py clean-contracts --from-json standardized_dataset/standardized_dataset.json
  py evaluate_limitations.py benchmark contract_dataset_ethereum --sizes 100 500 --fallback-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import filter_valid_solidity_files
from report_vulnerability_counts import aggregate_counts, load_functions_from_json
from standardize_dataset import DatasetStandardizer, FunctionData, parse_solidity_version_from_file


def _progress(message: str):
    print(message, flush=True)


def _group_functions_by_contract(functions: Iterable[FunctionData]) -> dict[str, list[FunctionData]]:
    grouped: dict[str, list[FunctionData]] = {}
    for fn in functions:
        grouped.setdefault(fn.contract_file, []).append(fn)
    return grouped


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def analyze_overflow_08(functions: list[FunctionData], sample_limit: int = 20) -> str:
    grouped = _group_functions_by_contract(functions)
    total_08_plus = 0
    flagged_08_plus = 0
    flagged_with_unchecked = 0
    suspicious_examples: list[str] = []

    total_contracts = len(grouped)
    _progress(f"[overflow-08] Checking {total_contracts} contracts...")
    for idx, (contract_file, contract_functions) in enumerate(grouped.items(), 1):
        if idx % 5000 == 0:
            _progress(f"[overflow-08] Processed {idx}/{total_contracts} contracts...")
        sol_version = parse_solidity_version_from_file(contract_file)
        if sol_version is None or sol_version < (0, 8):
            continue

        total_08_plus += 1
        has_overflow_label = any(
            "Integer Overflow/Underflow" in fn.vulnerabilities for fn in contract_functions
        )
        if not has_overflow_label:
            continue

        flagged_08_plus += 1
        source = _read_text(contract_file)
        if "unchecked" in source:
            flagged_with_unchecked += 1
        elif len(suspicious_examples) < sample_limit:
            suspicious_examples.append(contract_file)

    suspicious_no_unchecked = flagged_08_plus - flagged_with_unchecked
    lines = [
        "Solidity 0.8+ Overflow Analysis",
        "-" * 40,
        f"Contracts with pragma >=0.8.0: {total_08_plus}",
        f"0.8+ contracts labeled for Integer Overflow/Underflow: {flagged_08_plus}",
        f"0.8+ labeled contracts containing 'unchecked': {flagged_with_unchecked}",
        f"Potential false-positive candidates (0.8+ labeled, no 'unchecked'): {suspicious_no_unchecked}",
    ]
    if suspicious_examples:
        lines.append("")
        lines.append("Sample candidates:")
        lines.extend(f"  - {path}" for path in suspicious_examples)
    return "\n".join(lines)


def analyze_reentrancy_guards(functions: list[FunctionData], sample_limit: int = 20) -> str:
    flagged_functions = [fn for fn in functions if "Reentrancy" in fn.vulnerabilities]
    flagged_contracts = {fn.contract_file for fn in flagged_functions}
    _progress(f"[reentrancy-guards] Checking {len(flagged_functions)} labeled functions...")

    guarded_function_examples: list[str] = []
    guarded_functions = 0
    for idx, fn in enumerate(flagged_functions, 1):
        if idx % 10000 == 0:
            _progress(f"[reentrancy-guards] Scanned {idx}/{len(flagged_functions)} functions...")
        if "nonReentrant" in fn.function_code:
            guarded_functions += 1
            if len(guarded_function_examples) < sample_limit:
                guarded_function_examples.append(
                    f"{fn.contract_file} :: {fn.function_signature or fn.function_name}"
                )

    guarded_contracts = 0
    guarded_contract_examples: list[str] = []
    sorted_contracts = sorted(flagged_contracts)
    _progress(f"[reentrancy-guards] Checking {len(sorted_contracts)} source files for guards...")
    for idx, contract_file in enumerate(sorted_contracts, 1):
        if idx % 5000 == 0:
            _progress(f"[reentrancy-guards] Read {idx}/{len(sorted_contracts)} contracts...")
        source = _read_text(contract_file)
        if "nonReentrant" in source or "ReentrancyGuard" in source:
            guarded_contracts += 1
            if len(guarded_contract_examples) < sample_limit:
                guarded_contract_examples.append(contract_file)

    lines = [
        "Reentrancy Guard Analysis",
        "-" * 40,
        f"Functions labeled Reentrancy: {len(flagged_functions)}",
        f"Contracts containing at least one Reentrancy label: {len(flagged_contracts)}",
        f"Labeled functions whose code includes 'nonReentrant': {guarded_functions}",
        f"Labeled contracts mentioning 'nonReentrant' or 'ReentrancyGuard': {guarded_contracts}",
    ]
    if guarded_function_examples:
        lines.append("")
        lines.append("Sample guarded functions:")
        lines.extend(f"  - {item}" for item in guarded_function_examples)
    if guarded_contract_examples:
        lines.append("")
        lines.append("Sample guarded contracts:")
        lines.extend(f"  - {item}" for item in guarded_contract_examples)
    return "\n".join(lines)


def analyze_clean_contracts(functions: list[FunctionData], sample_limit: int = 20) -> str:
    _progress("[clean-contracts] Aggregating clean-contract statistics...")
    stats = aggregate_counts(functions)
    clean_contracts = sorted(stats.get("contract_files_with_no_vulns", []))
    total = stats.get("total_contracts", 0)
    clean = stats.get("contracts_with_no_vulns", 0)
    pct = (100.0 * clean / total) if total else 0.0

    lines = [
        "Clean Contract Analysis",
        "-" * 40,
        f"Total valid contracts: {total}",
        f"Contracts with no vulnerabilities: {clean}",
        f"Clean-contract percentage: {pct:.2f}%",
    ]
    if clean_contracts:
        lines.append("")
        lines.append("Sample clean contracts:")
        lines.extend(f"  - {path}" for path in clean_contracts[:sample_limit])
    return "\n".join(lines)


def benchmark_directory(
    directory: str,
    sizes: list[int],
    fallback_only: bool,
    validate: bool,
    skip_duplicates: bool,
) -> str:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")

    all_paths = list(root.rglob("*.sol"))
    if not all_paths:
        raise FileNotFoundError(f"No .sol files found in {directory}")

    lines = [
        "Scalability Benchmark",
        "-" * 40,
        f"Directory: {root}",
        f"Total discovered .sol files: {len(all_paths)}",
        f"fallback_only={fallback_only}, validate={validate}, skip_duplicates={skip_duplicates}",
        "",
        "size | kept_files | functions | failed_files | seconds | contracts_per_min",
        "-" * 72,
    ]

    for size in sizes:
        _progress(f"[benchmark] Preparing subset size {size}...")
        subset = all_paths[:size]
        if validate or skip_duplicates:
            kept_paths, _ = filter_valid_solidity_files(
                subset,
                validate=validate,
                skip_duplicates=skip_duplicates,
                validation_min_length=50,
            )
        else:
            kept_paths = [Path(p).resolve() for p in subset]

        standardizer = DatasetStandardizer(
            output_dir="benchmark_output",
            fallback_only=fallback_only,
        )

        start = time.perf_counter()
        failed_files = 0
        for idx, sol_file in enumerate(kept_paths, 1):
            if idx % 100 == 0:
                _progress(f"[benchmark] Size {size}: processed {idx}/{len(kept_paths)} files...")
            try:
                standardizer.all_functions.extend(standardizer.process_file(str(sol_file)))
            except Exception:
                failed_files += 1
        elapsed = time.perf_counter() - start
        rate = (len(kept_paths) / elapsed * 60.0) if elapsed else 0.0

        lines.append(
            f"{size:4} | {len(kept_paths):10} | {len(standardizer.all_functions):9} | "
            f"{failed_files:12} | {elapsed:7.2f} | {rate:17.2f}"
        )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Empirical evaluation helpers for labeling/tool limitations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("summary", "overflow-08", "reentrancy-guards", "clean-contracts"):
        p = subparsers.add_parser(name)
        p.add_argument("--from-json", required=True, help="Path to standardized_dataset.json")
        p.add_argument("-o", "--output", help="Optional report output file")
        p.add_argument("--sample-limit", type=int, default=20, help="Sample paths to print")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("directory", help="Directory containing .sol files")
    benchmark.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 1000])
    benchmark.add_argument("--fallback-only", action="store_true")
    benchmark.add_argument("--no-validate", action="store_true")
    benchmark.add_argument("--no-dedup", action="store_true")
    benchmark.add_argument("-o", "--output", help="Optional report output file")

    return parser


def _write_output(text: str, output: str | None):
    print(text)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
        print(f"\nReport saved to {output}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "benchmark":
        _progress("[benchmark] Starting benchmark...")
        report = benchmark_directory(
            args.directory,
            sizes=args.sizes,
            fallback_only=args.fallback_only,
            validate=not args.no_validate,
            skip_duplicates=not args.no_dedup,
        )
        _write_output(report, args.output)
        return

    _progress(f"[{args.command}] Loading JSON from {args.from_json}...")
    functions = load_functions_from_json(args.from_json)
    if not functions:
        print(f"No functions loaded from {args.from_json}")
        sys.exit(1)
    _progress(f"[{args.command}] Loaded {len(functions)} functions.")

    reports = []
    if args.command in ("summary", "overflow-08"):
        _progress("[summary] Running Solidity 0.8+ overflow analysis..." if args.command == "summary" else "[overflow-08] Running analysis...")
        reports.append(analyze_overflow_08(functions, sample_limit=args.sample_limit))
    if args.command in ("summary", "reentrancy-guards"):
        _progress("[summary] Running reentrancy guard analysis..." if args.command == "summary" else "[reentrancy-guards] Running analysis...")
        reports.append(analyze_reentrancy_guards(functions, sample_limit=args.sample_limit))
    if args.command in ("summary", "clean-contracts"):
        _progress("[summary] Running clean-contract analysis..." if args.command == "summary" else "[clean-contracts] Running analysis...")
        reports.append(analyze_clean_contracts(functions, sample_limit=args.sample_limit))

    _write_output("\n\n".join(reports), args.output)


if __name__ == "__main__":
    main()
