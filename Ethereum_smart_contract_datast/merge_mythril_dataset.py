#!/usr/bin/env python3
"""
Merge Mythril scan results with the primary standardized dataset.

Produces a combined dataset that includes:
- Primary: pattern-based labels (Reentrancy, Timestamp, Integer Overflow, Delegatecall)
- Supplement: Mythril labels (adds TOD, Uninitialized Storage, Unchecked Call, etc.)

Usage:
  py merge_mythril_dataset.py --primary standardized_dataset/standardized_dataset.json --mythril mythril_scan_output/mythril_scan_results.json -o combined_dataset.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mythril_by_file(mythril_results: Dict) -> Dict[str, List[Dict]]:
    """Map file path -> list of Mythril issues for that file."""
    by_file = {}
    for c in mythril_results.get("contracts", []):
        fp = c.get("file", "")
        if fp:
            by_file[fp] = c.get("issues", [])
    return by_file


def merge(
    primary_path: str,
    mythril_path: str,
    output_path: str,
) -> None:
    """
    Merge primary dataset (function-level) with Mythril (contract-level).
    For each function, add Mythril issues from its contract file.
    """
    primary = load_json(primary_path)
    mythril = load_json(mythril_path)
    mythril_by_file = build_mythril_by_file(mythril)

    functions = primary.get("functions", [])
    for func in functions:
        cf = func.get("contract_file", "")
        m_issues = mythril_by_file.get(cf, [])
        if m_issues:
            existing_swc = set(func.get("swc_ids", []))
            for i in m_issues:
                sid = i.get("swc_id") or i.get("title", "")
                if sid and sid not in existing_swc:
                    existing_swc.add(sid)
                    func.setdefault("mythril_issues", []).append(i)
            func["swc_ids"] = list(existing_swc)
            func["vulnerabilities"] = list(set(
                func.get("vulnerabilities", []) +
                [i.get("title", "") for i in m_issues if i.get("title")]
            ))

    primary["metadata"] = primary.get("metadata", {})
    primary["metadata"]["merged_with_mythril"] = True
    primary["metadata"]["mythril_source"] = mythril_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(primary, f, indent=2, ensure_ascii=False)
    print(f"Merged dataset saved to {output_path}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Merge Mythril results with primary dataset")
    p.add_argument("--primary", "-p", required=True, help="Path to standardized_dataset.json")
    p.add_argument("--mythril", "-m", required=True, help="Path to mythril_scan_results.json")
    p.add_argument("-o", "--output", default="combined_dataset.json", help="Output path")
    args = p.parse_args()
    merge(args.primary, args.mythril, args.output)


if __name__ == "__main__":
    main()
