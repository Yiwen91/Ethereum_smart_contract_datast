#!/usr/bin/env python3
"""
Test that vulnerability labeling runs successfully.

Run from project root:
  python test_labeling.py

Checks:
  1. Unit tests: known code snippets get expected labels.
  2. Integration: processing a .sol file yields FunctionData with labels and SWC IDs.
  3. Optional: run on a real contract path and print a labeling report.
"""

import sys
from pathlib import Path

# Allow running from project root or from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from standardize_dataset import VulnerabilityLabeler, DatasetStandardizer, FunctionData, SWC_MAPPING


def test_timestamp_labeling():
    """Code with now/block.timestamp should get timestamp labels; rule: TimestampInvoc ∧ (Assign ∨ Contaminate)."""
    labeler = VulnerabilityLabeler()

    # Must have TimestampInvoc (uses now/block.timestamp)
    code_invoc = "function f() view returns (uint) { return uint(now) % 10; }"
    vulns, labels = labeler.label_function(code_invoc)
    assert labels.get("TimestampInvoc") is True, "Expected TimestampInvoc true when using now"

    # Contaminate: timestamp in condition (e.g. now > x)
    code_contaminate = """
    function lock() public {
        if (now > deadline) revert();
        locked = true;
    }
    """
    vulns2, labels2 = labeler.label_function(code_contaminate)
    assert labels2.get("TimestampInvoc") is True and labels2.get("TimestampContaminate") is True, (
        "Expected TimestampInvoc and TimestampContaminate for condition with now"
    )
    assert "Timestamp Dependency" in vulns2, "Expected Timestamp Dependency when Invoc and Contaminate"
    return True


def test_reentrancy_labeling():
    """Code with .transfer() or .call() should be labeled Reentrancy; NOT if nonReentrant/ReentrancyGuard."""
    labeler = VulnerabilityLabeler()

    code_transfer = """
    function withdraw() public {
        uint amt = balances[msg.sender];
        msg.sender.transfer(amt);
        balances[msg.sender] = 0;
    }
    """
    vulns, labels = labeler.label_function(code_transfer)
    assert "Reentrancy" in vulns, "Expected Reentrancy for .transfer() without guard"
    assert labels.get("Reentrancy") is True

    # With nonReentrant modifier - should NOT be labeled (avoid over-labeling)
    code_guarded = """
    function withdraw() public nonReentrant {
        uint amt = balances[msg.sender];
        msg.sender.transfer(amt);
        balances[msg.sender] = 0;
    }
    """
    vulns2, _ = labeler.label_function(code_guarded)
    assert "Reentrancy" not in vulns2, "Should NOT label Reentrancy when nonReentrant modifier present"
    return True


def test_integer_overflow_labeling():
    """Code with ++ or += should get Integer Overflow; NOT for Solidity 0.8+ (native protection)."""
    labeler = VulnerabilityLabeler()

    code_inc = """
    function tick() public {
        counter++;
        total += 1;
    }
    """
    vulns, labels = labeler.label_function(code_inc)  # no sol_version -> pre-0.8
    assert "Integer Overflow/Underflow" in vulns, "Expected Integer Overflow for pre-0.8"
    assert labels.get("IntegerOverflow") is True

    # Solidity 0.8+ has native overflow protection - skip labeling
    vulns2, _ = labeler.label_function(code_inc, sol_version=(0, 8))
    assert "Integer Overflow/Underflow" not in vulns2, "Should NOT label overflow for Solidity 0.8+"
    return True


def test_delegatecall_labeling():
    """Code with delegatecall should get Dangerous Delegatecall."""
    labeler = VulnerabilityLabeler()

    code_del = """
    function run(address target, bytes memory data) public {
        (bool ok,) = target.delegatecall(data);
        require(ok);
    }
    """
    vulns, labels = labeler.label_function(code_del)
    assert "Dangerous Delegatecall" in vulns, "Expected Dangerous Delegatecall"
    assert labels.get("Delegatecall") is True
    return True


def test_swc_mapping():
    """Every vulnerability type should map to an SWC ID."""
    for vuln_name, info in SWC_MAPPING.items():
        assert "swc_id" in info, f"Missing swc_id for {vuln_name}"
        assert info["swc_id"].startswith("SWC-"), f"Invalid swc_id for {vuln_name}: {info['swc_id']}"
    return True


def test_process_file_yields_labels():
    """Processing a real .sol file should return FunctionData with labels and swc_ids."""
    standardizer = DatasetStandardizer(output_dir="test_output")
    # Use a path that exists in the repo if available
    candidates = [
        Path(__file__).parent / "contract_dataset_ethereum" / "contract1" / "0.sol",
        Path("contract_dataset_ethereum/contract1/0.sol"),
        Path("Ethereum_smart_contract_datast/contract_dataset_ethereum/contract1/0.sol"),
    ]
    sol_file = None
    for p in candidates:
        if p.exists():
            sol_file = str(p)
            break

    if not sol_file:
        print("  [SKIP] No sample .sol file found for integration test")
        return True

    functions = standardizer.process_file(sol_file)
    assert len(functions) > 0, "Expected at least one function from sample contract"

    for func in functions:
        assert isinstance(func, FunctionData), "Expected FunctionData"
        assert hasattr(func, "vulnerabilities") and isinstance(func.vulnerabilities, list), (
            "Expected vulnerabilities list"
        )
        assert hasattr(func, "swc_ids") and isinstance(func.swc_ids, list), "Expected swc_ids list"
        assert hasattr(func, "labels") and isinstance(func.labels, dict), "Expected labels dict"
        # Each listed vulnerability should have an SWC ID
        for v in func.vulnerabilities:
            assert v in SWC_MAPPING, f"Unknown vulnerability type: {v}"

    # Sample contract has now/transfer/++ so we expect at least one function with some label
    any_label = any(
        (f.vulnerabilities or f.labels) for f in functions
    )
    assert any_label, "Expected at least one function with vulnerabilities or non-empty labels"
    return True


def run_all_tests():
    tests = [
        ("Timestamp labeling", test_timestamp_labeling),
        ("Reentrancy labeling", test_reentrancy_labeling),
        ("Integer overflow labeling", test_integer_overflow_labeling),
        ("Delegatecall labeling", test_delegatecall_labeling),
        ("SWC mapping", test_swc_mapping),
        ("Process file yields labels", test_process_file_yields_labels),
    ]
    failed = []
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  OK: {name}")
        except AssertionError as e:
            print(f"  FAIL: {name} — {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR: {name} — {e}")
            failed.append(name)

    return failed


def print_labeling_report(sol_file: str):
    """Print a short report showing that labeling ran successfully for one file."""
    standardizer = DatasetStandardizer(output_dir="test_output")
    path = Path(sol_file)
    if not path.exists():
        print(f"File not found: {sol_file}")
        return False

    functions = standardizer.process_file(sol_file)
    print(f"File: {sol_file}")
    print(f"Functions: {len(functions)}")
    print("-" * 60)
    for f in functions:
        vuln_str = ", ".join(f.vulnerabilities) if f.vulnerabilities else "none"
        swc_str = ", ".join(f.swc_ids) if f.swc_ids else "-"
        print(f"  {f.function_name}: vulnerabilities=[{vuln_str}]  swc_ids=[{swc_str}]")
        if f.labels:
            true_labels = [k for k, v in f.labels.items() if v]
            if true_labels:
                print(f"    labels: {true_labels}")
    print("-" * 60)
    print("Labeling completed successfully.")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test that labeling works")
    parser.add_argument("--report", type=str, metavar="FILE", help="Print labeling report for this .sol file")
    args = parser.parse_args()

    if args.report:
        ok = print_labeling_report(args.report)
        sys.exit(0 if ok else 1)
    else:
        print("Running labeling tests...")
        failed = run_all_tests()
        if failed:
            print(f"\nFailed: {failed}")
            sys.exit(1)
        print("\nAll tests passed — labeling is working successfully.")
        sys.exit(0)
