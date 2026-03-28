#!/usr/bin/env python3
r"""
Create reproducible train/validation/test splits from a standardized dataset JSON.

Goals:
1. Keep all functions from the same contract in the same split.
2. Balance the 7 vulnerability types as evenly as possible across splits.
3. Save split-specific standardized JSON files plus a readable summary report.

Default use case:
  py prepare_experiment_splits.py --from-json standardized_dataset\standardized_dataset.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from report_vulnerability_counts import VULN_TYPES, load_functions_from_json


def _group_functions_by_contract(functions: list) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for fn in functions:
        grouped[fn.contract_file].append(fn)
    return grouped


def _largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    if remainder <= 0:
        return counts

    ranked = sorted(
        ratios.keys(),
        key=lambda name: (raw[name] - counts[name], raw[name], name),
        reverse=True,
    )
    for idx in range(remainder):
        counts[ranked[idx % len(ranked)]] += 1
    return counts


def _contract_vulnerability_set(contract_functions: list) -> tuple[str, ...]:
    labels = set()
    for fn in contract_functions:
        for vuln in fn.vulnerabilities:
            if vuln in VULN_TYPES:
                labels.add(vuln)
    return tuple(sorted(labels))


def _split_contracts_by_labelset(
    grouped_contracts: dict[str, list],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    label_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for contract_file, contract_functions in grouped_contracts.items():
        label_groups[_contract_vulnerability_set(contract_functions)].append(contract_file)

    rng = random.Random(seed)
    split_contracts = {name: [] for name in ratios}

    for label_key, contracts in sorted(
        label_groups.items(),
        key=lambda item: (len(item[0]), item[0], len(item[1])),
        reverse=True,
    ):
        shuffled = list(contracts)
        rng.shuffle(shuffled)
        counts = _largest_remainder_counts(len(shuffled), ratios)

        start = 0
        for split_name in ratios:
            end = start + counts[split_name]
            split_contracts[split_name].extend(shuffled[start:end])
            start = end

    return split_contracts


def _build_split_functions(grouped_contracts: dict[str, list], split_contracts: dict[str, list[str]]) -> dict[str, list]:
    return {
        split_name: [
            fn
            for contract_file in contract_files
            for fn in grouped_contracts[contract_file]
        ]
        for split_name, contract_files in split_contracts.items()
    }


def _vulnerability_contract_counts(grouped_contracts: dict[str, list], contract_files: list[str]) -> dict[str, int]:
    counts = {vuln: 0 for vuln in VULN_TYPES}
    for contract_file in contract_files:
        labels = set(_contract_vulnerability_set(grouped_contracts[contract_file]))
        for vuln in labels:
            counts[vuln] += 1
    return counts


def _vulnerability_function_counts(functions: list) -> dict[str, int]:
    counts = {vuln: 0 for vuln in VULN_TYPES}
    for fn in functions:
        for vuln in fn.vulnerabilities:
            if vuln in counts:
                counts[vuln] += 1
    return counts


def _format_summary(
    grouped_contracts: dict[str, list],
    split_contracts: dict[str, list[str]],
    split_functions: dict[str, list],
    ratios: dict[str, float],
    seed: int,
) -> str:
    total_contracts = len(grouped_contracts)
    total_functions = sum(len(items) for items in grouped_contracts.values())
    global_contract_counts = _vulnerability_contract_counts(grouped_contracts, list(grouped_contracts.keys()))

    lines = [
        "Experiment Split Summary",
        "=" * 72,
        f"Seed: {seed}",
        f"Ratios: train={ratios['train']:.2f}, val={ratios['val']:.2f}, test={ratios['test']:.2f}",
        f"Total contracts: {total_contracts}",
        f"Total functions: {total_functions}",
        "",
        "Per split overview",
        "-" * 72,
    ]

    for split_name in ("train", "val", "test"):
        contract_files = split_contracts[split_name]
        functions = split_functions[split_name]
        lines.append(
            f"{split_name:5} contracts={len(contract_files):6} "
            f"functions={len(functions):8} "
            f"ratio={((len(contract_files) / total_contracts) if total_contracts else 0.0):.4f}"
        )

    lines.extend([
        "",
        "Per vulnerability contract balance",
        "-" * 72,
    ])

    header = (
        f"{'Vulnerability':30} {'global':>8} "
        f"{'train':>8} {'val':>8} {'test':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for vuln in VULN_TYPES:
        row = [f"{vuln:30}", f"{global_contract_counts[vuln]:8d}"]
        for split_name in ("train", "val", "test"):
            split_count = _vulnerability_contract_counts(grouped_contracts, split_contracts[split_name])[vuln]
            global_count = global_contract_counts[vuln]
            pct = (100.0 * split_count / global_count) if global_count else 0.0
            row.append(f"{split_count:4d} ({pct:5.1f}%)")
        lines.append(" ".join(row))

    lines.extend([
        "",
        "Per vulnerability function counts",
        "-" * 72,
    ])

    header = (
        f"{'Vulnerability':30} {'train':>8} {'val':>8} {'test':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    split_function_counts = {
        split_name: _vulnerability_function_counts(functions)
        for split_name, functions in split_functions.items()
    }
    for vuln in VULN_TYPES:
        lines.append(
            f"{vuln:30} "
            f"{split_function_counts['train'][vuln]:8d} "
            f"{split_function_counts['val'][vuln]:8d} "
            f"{split_function_counts['test'][vuln]:8d}"
        )

    return "\n".join(lines) + "\n"


def _write_split_json(
    output_path: Path,
    functions: list,
    source_json: str,
    split_name: str,
    seed: int,
    ratios: dict[str, float],
):
    payload = {
        "metadata": {
            "source_json": source_json,
            "split_name": split_name,
            "seed": seed,
            "ratios": ratios,
            "total_functions": len(functions),
            "total_contracts": len({fn.contract_file for fn in functions}),
            "vulnerability_types": VULN_TYPES,
            "format_version": "1.0",
        },
        "functions": [
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
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_splits(
    from_json: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> str:
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be > 0.")

    total_ratio = train_ratio + val_ratio + test_ratio
    ratios = {
        "train": train_ratio / total_ratio,
        "val": val_ratio / total_ratio,
        "test": test_ratio / total_ratio,
    }

    functions = load_functions_from_json(from_json)
    if not functions:
        raise ValueError(f"No functions loaded from {from_json}")

    grouped_contracts = _group_functions_by_contract(functions)
    split_contracts = _split_contracts_by_labelset(grouped_contracts, ratios, seed)
    split_functions = _build_split_functions(grouped_contracts, split_contracts)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        contracts_file = out_dir / f"{split_name}_contracts.txt"
        contracts_file.write_text(
            "\n".join(sorted(split_contracts[split_name])) + "\n",
            encoding="utf-8",
        )
        _write_split_json(
            out_dir / f"{split_name}.json",
            split_functions[split_name],
            source_json=str(Path(from_json).resolve()),
            split_name=split_name,
            seed=seed,
            ratios=ratios,
        )

    manifest = {
        "source_json": str(Path(from_json).resolve()),
        "seed": seed,
        "ratios": ratios,
        "splits": {
            split_name: {
                "contracts": len(split_contracts[split_name]),
                "functions": len(split_functions[split_name]),
                "contracts_file": str((out_dir / f"{split_name}_contracts.txt").resolve()),
                "json_file": str((out_dir / f"{split_name}.json").resolve()),
            }
            for split_name in ("train", "val", "test")
        },
        "vulnerability_types": VULN_TYPES,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    summary = _format_summary(grouped_contracts, split_contracts, split_functions, ratios, seed)
    (out_dir / "split_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create balanced train/val/test splits from standardized_dataset.json"
    )
    parser.add_argument(
        "--from-json",
        required=True,
        help="Path to standardized_dataset.json",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_splits/esc_primary",
        help="Output directory for split files",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    summary = create_splits(
        from_json=args.from_json,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(summary, end="")
    print(f"Saved split files to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
