#!/usr/bin/env python3
"""
Example usage of the dataset standardization tool and helper functions.
"""

from pathlib import Path
from collections import defaultdict

from standardize_dataset import DatasetStandardizer

# Helpers: validation and duplicate detection
try:
    from helpers import (
        validate_solidity_file,
        validate_solidity_content,
        normalize_solidity_for_dedup,
        compute_content_hash,
        find_duplicate_files,
        get_duplicate_groups,
        filter_valid_solidity_files,
        choose_canonical_from_group,
        find_structural_duplicates,
    )
    HAS_HELPERS = True
except ImportError:
    HAS_HELPERS = False

def example_single_file():
    """Example: Process a single .sol file"""
    standardizer = DatasetStandardizer(output_dir="output_example")
    
    # Process a single file
    sol_file = "Ethereum_smart_contract_datast/contract_dataset_ethereum/contract1/0.sol"
    
    if Path(sol_file).exists():
        functions = standardizer.process_file(sol_file)
        print(f"Extracted {len(functions)} functions from {sol_file}")
        
        # Export results
        standardizer.all_functions = functions
        standardizer.export_json("example_output.json")
        standardizer.export_csv("example_output.csv")
        
        # Print some statistics
        for func in functions:
            if func.vulnerabilities:
                print(f"\nFunction: {func.function_name}")
                print(f"  Vulnerabilities: {func.vulnerabilities}")
                print(f"  SWC IDs: {func.swc_ids}")
                print(f"  Labels: {func.labels}")
    else:
        print(f"File not found: {sol_file}")

def example_directory():
    """Example: Process a directory"""
    standardizer = DatasetStandardizer(output_dir="output_directory")
    
    # Process a directory
    input_dir = "Ethereum_smart_contract_datast/contract_dataset_ethereum/contract1"
    
    if Path(input_dir).exists():
        standardizer.process_directory(input_dir, recursive=False)
        
        # Export results
        standardizer.export_json()
        standardizer.export_csv()
        
        print(f"\nProcessed {len(standardizer.all_functions)} functions")
        
        # Print vulnerability statistics
        from collections import defaultdict
        vuln_counts = defaultdict(int)
        for func in standardizer.all_functions:
            for vuln in func.vulnerabilities:
                vuln_counts[vuln] += 1
        
        print("\nVulnerability Statistics:")
        for vuln, count in sorted(vuln_counts.items()):
            print(f"  {vuln}: {count}")
    else:
        print(f"Directory not found: {input_dir}")

def example_validation():
    """Example: Solidity validation (filter invalid contracts)"""
    if not HAS_HELPERS:
        print("Helpers not available, skipping validation example.")
        return

    # Validate a file
    sol_file = Path("Ethereum_smart_contract_datast/contract_dataset_ethereum/contract1/0.sol")
    if sol_file.exists():
        result = validate_solidity_file(sol_file)
        print(f"Validation: valid={result.valid}, errors={result.errors}, warnings={result.warnings}")
    else:
        print("Sample file not found, using inline content.")
        code = "pragma solidity ^0.8.0; contract C { function f() public {} }"
        result = validate_solidity_content(code)
        print(f"Validation: valid={result.valid}, summary={result.summary}")

    # Filter a list of files: keep only valid, skip duplicates
    input_dir = Path("Ethereum_smart_contract_datast/contract_dataset_ethereum/contract1")
    if input_dir.exists():
        paths = list(input_dir.glob("*.sol"))
        kept, stats = filter_valid_solidity_files(
            paths, validate=True, skip_duplicates=True
        )
        print(f"Filter stats: total={stats.total}, valid={stats.valid}, invalid={stats.invalid}, "
              f"duplicate_skipped={stats.duplicate_skipped}, duplicate_groups={stats.duplicate_groups}")


def example_duplicate_detection():
    """Example: Duplicate detection (content hash and structural)"""
    if not HAS_HELPERS:
        print("Helpers not available, skipping duplicate detection example.")
        return

    input_dir = Path("Ethereum_smart_contract_datast/contract_dataset_ethereum/contract1")
    if not input_dir.exists():
        print("Sample directory not found.")
        return

    paths = list(input_dir.glob("*.sol"))
    if not paths:
        print("No .sol files in directory.")
        return

    # Content-based duplicates (same source after normalizing comments/whitespace)
    dup_groups = get_duplicate_groups(paths, normalize=True)
    print(f"Content-duplicate groups: {len(dup_groups)}")
    for i, group in enumerate(dup_groups[:3]):
        canonical = choose_canonical_from_group(group, prefer_short_path=True)
        print(f"  Group {i+1}: keep {canonical}, duplicates: {[p for p in group if p != canonical]}")

    # Structural duplicates (same contract/function names, possibly different code)
    struct_dups = find_structural_duplicates(paths)
    print(f"Structural-duplicate groups: {len(struct_dups)}")


if __name__ == "__main__":
    print("Example 1: Processing a single file")
    print("=" * 50)
    example_single_file()

    print("\n\nExample 2: Processing a directory")
    print("=" * 50)
    example_directory()

    if HAS_HELPERS:
        print("\n\nExample 3: Validation and filtering")
        print("=" * 50)
        example_validation()

        print("\n\nExample 4: Duplicate detection")
        print("=" * 50)
        example_duplicate_detection()
