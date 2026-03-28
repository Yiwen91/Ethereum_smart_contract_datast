#!/usr/bin/env python3
"""
Standardized Dataset Format Processor for Ethereum Smart Contracts

This script:
1. Extracts functions from .sol files using Slither
2. Applies exact labeling rules for vulnerabilities
3. Maps labels to SWC IDs
4. Outputs standardized JSON/CSV format
"""

import os
import json
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from helpers import (
        validate_solidity_file,
        validate_solidity_content,
        filter_valid_solidity_files,
        find_duplicate_files,
        get_duplicate_groups,
        FilterStats,
    )
except ImportError:
    validate_solidity_file = None
    validate_solidity_content = None
    filter_valid_solidity_files = None
    find_duplicate_files = None
    get_duplicate_groups = None
    FilterStats = None

# SWC ID Mapping (can be loaded from swc_mapping.json)
SWC_MAPPING = {
    "Reentrancy": {
        "swc_id": "SWC-107",
        "swc_name": "Reentrancy"
    },
    "Timestamp Dependency": {
        "swc_id": "SWC-116",
        "swc_name": "Block Timestamp Dependence"
    },
    "Integer Overflow/Underflow": {
        "swc_id": "SWC-101",
        "swc_name": "Integer Overflow and Underflow"
    },
    "Dangerous Delegatecall": {
        "swc_id": "SWC-112",
        "swc_name": "Delegatecall to Untrusted Contract"
    },
    "Transaction-Ordering Dependence": {
        "swc_id": "SWC-114",
        "swc_name": "Transaction-Ordering Dependence"
    },
    "Uninitialized Storage Pointer": {
        "swc_id": "SWC-109",
        "swc_name": "Uninitialized Storage Pointer"
    },
    "Unchecked External Calls": {
        "swc_id": "SWC-104",
        "swc_name": "Unchecked Call Return Value"
    }
}

def parse_solidity_version_from_file(sol_file: str) -> Optional[Tuple[int, int]]:
    """
    Parse pragma solidity version from a .sol file.
    Returns (major, minor) e.g. (0, 8) for ^0.8.0, or None if not found.
    """
    try:
        content = Path(sol_file).read_text(encoding="utf-8", errors="replace")
        m = re.search(r'pragma\s+solidity\s+([^;]+);', content, re.IGNORECASE)
        if not m:
            return None
        pragma = m.group(1).strip()
        # Extract first version number: ^0.8.0, >=0.7.0 <0.9.0, 0.8.0, etc.
        version_m = re.search(r'(\d+)\.(\d+)(?:\.\d+)?', pragma)
        if version_m:
            return (int(version_m.group(1)), int(version_m.group(2)))
    except Exception:
        pass
    return None


def load_swc_mapping(mapping_file: str = "swc_mapping.json") -> Dict:
    """Load SWC mapping from JSON file if available"""
    try:
        mapping_path = Path(__file__).parent / mapping_file
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                data = json.load(f)
                return data.get('SWC_MAPPING', SWC_MAPPING)
    except Exception as e:
        print(f"Warning: Could not load SWC mapping from {mapping_file}: {e}")
    return SWC_MAPPING


@dataclass
class FunctionData:
    """Standardized function data structure"""
    contract_file: str
    contract_name: str
    function_name: str
    function_signature: str
    function_code: str
    start_line: int
    end_line: int
    visibility: str
    state_mutability: str
    vulnerabilities: List[str]  # List of vulnerability types detected
    swc_ids: List[str]  # List of SWC IDs
    labels: Dict[str, bool]  # Detailed labels (e.g., TimestampInvoc, TimestampAssign, etc.)
    metadata: Dict  # Additional metadata


class VulnerabilityLabeler:
    """Applies exact labeling rules for vulnerability detection"""
    
    def __init__(self):
        # Patterns for timestamp dependency detection
        self.timestamp_patterns = {
            'invoc': [r'\bnow\b', r'\bblock\.timestamp\b', r'\bblock\.number\b'],
            'assign': [r'\bnow\s*=', r'block\.timestamp\s*=', r'block\.number\s*='],
            'contaminate': [r'\bnow\b', r'\bblock\.timestamp\b']  # Used in conditions/operations
        }
        
        # Patterns for reentrancy detection
        self.reentrancy_patterns = {
            'external_call': [r'\.call\s*\(', r'\.send\s*\(', r'\.transfer\s*\(', 
                            r'\.delegatecall\s*\(', r'\.callcode\s*\('],
            'state_change_after': r'(\.call|\.send|\.transfer|\.delegatecall|\.callcode)'
        }
        
        # Patterns for integer overflow/underflow
        self.integer_overflow_patterns = [
            r'\+\+', r'--', r'\+\s*=', r'-\s*=', r'\*\s*=', r'/\s*=',
            r'uint\d*\s+\w+\s*[+\-*/]', r'int\d*\s+\w+\s*[+\-*/]'
        ]
        
        # Patterns for delegatecall
        self.delegatecall_patterns = [
            r'\.delegatecall\s*\(',
            r'delegatecall\s*\('
        ]

        # Patterns for order-dependent shared state and bidding-like operations
        self.tod_state_patterns = [
            r'\b(highestBid|highestbid|bid|bids|auction|price|nonce|order|orders|winner|pending)\b',
            r'\b(balanceOf|allowance|balances?)\b',
            r'\[(.*?)\]\s*=',
        ]

        # Patterns for unchecked low-level calls
        self.unchecked_external_call_patterns = [
            r'(?P<expr>[^;\n]*\.(?:call|send|delegatecall|callcode)\s*\([^;\n]*\))\s*;',
        ]
    
    def detect_timestamp_dependency(self, code: str) -> Dict[str, bool]:
        """
        Detects timestamp dependency using rule:
        TimestampInvoc ∧ (TimestampAssign ∨ TimestampContaminate)
        """
        labels = {
            'TimestampInvoc': False,
            'TimestampAssign': False,
            'TimestampContaminate': False
        }
        
        # Check for timestamp invocation
        for pattern in self.timestamp_patterns['invoc']:
            if re.search(pattern, code, re.IGNORECASE):
                labels['TimestampInvoc'] = True
                break
        
        # Check for timestamp assignment
        for pattern in self.timestamp_patterns['assign']:
            if re.search(pattern, code, re.IGNORECASE):
                labels['TimestampAssign'] = True
                break
        
        # Check for timestamp contamination (used in conditions, operations)
        # This is more complex - we look for timestamp in conditional/logical operations
        for pattern in self.timestamp_patterns['contaminate']:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                # Check if it's used in a condition or operation (not just assignment)
                context_start = max(0, match.start() - 50)
                context_end = min(len(code), match.end() + 50)
                context = code[context_start:context_end]
                
                # Check for conditional operators
                if re.search(r'[=<>!&|]', context):
                    labels['TimestampContaminate'] = True
                    break
        
        return labels
    
    def _has_reentrancy_guard(self, code: str) -> bool:
        """
        Check if function has ReentrancyGuard or nonReentrant modifier (common fix).
        If present, the function is protected and should NOT be labeled as reentrant.
        """
        # Modifiers appear in the declaration (before the first { of the body)
        decl_end = code.find('{')
        declaration = code[:decl_end] if decl_end >= 0 else code
        return bool(re.search(r'\b(nonReentrant|ReentrancyGuard|reentrancyGuard)\b', declaration, re.IGNORECASE))

    def detect_reentrancy(self, code: str) -> bool:
        """
        Detects reentrancy vulnerability.
        Skips labeling if function has ReentrancyGuard/nonReentrant modifier (#1 cause of over-labeling).
        """
        if self._has_reentrancy_guard(code):
            return False
        has_external_call = False
        for pattern in self.reentrancy_patterns['external_call']:
            if re.search(pattern, code, re.IGNORECASE):
                has_external_call = True
                break
        if has_external_call:
            return True
        return False
    
    def detect_integer_overflow(self, code: str) -> bool:
        """Detects potential integer overflow/underflow"""
        for pattern in self.integer_overflow_patterns:
            if re.search(pattern, code):
                return True
        return False
    
    def detect_delegatecall(self, code: str) -> bool:
        """Detects dangerous delegatecall usage"""
        for pattern in self.delegatecall_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def detect_transaction_ordering_dependence(self, code: str) -> bool:
        """
        Detects likely transaction-ordering dependence / front-running patterns.
        Heuristic: public/external function touches shared order-sensitive state and
        also performs a comparison/update that can be influenced by transaction ordering.
        """
        declaration = code[: code.find('{')] if '{' in code else code
        if not re.search(r'\b(public|external)\b', declaration, re.IGNORECASE):
            return False

        has_order_state = any(re.search(pattern, code, re.IGNORECASE) for pattern in self.tod_state_patterns)
        if not has_order_state:
            return False

        has_competitive_logic = bool(re.search(r'(>=|<=|>|<|==|!=)', code))
        has_value_flow = bool(
            re.search(r'\b(msg\.value|transfer\s*\(|send\s*\(|call\s*\(|approve\s*\(|transferFrom\s*\()', code, re.IGNORECASE)
        )
        has_state_write = bool(re.search(r'(\w+\s*[\+\-*/]?=|\[[^\]]+\]\s*=)', code))
        return has_state_write and (has_competitive_logic or has_value_flow)

    def detect_uninitialized_storage_pointer(self, code: str) -> bool:
        """
        Detects uninitialized local storage pointers such as:
        `uint[] storage s;` or `MyStruct storage data;`
        """
        declaration = code[: code.find('{')] if '{' in code else code
        body = code[code.find('{') + 1 :] if '{' in code else ""
        if re.search(r'\b(storage)\b', declaration, re.IGNORECASE):
            return False
        pattern = r'\b(?:mapping\s*\([^;]+?\)|[A-Za-z_]\w*(?:\[\])*)\s+storage\s+[A-Za-z_]\w*\s*;'
        return bool(re.search(pattern, body, re.IGNORECASE))

    def detect_unchecked_external_calls(self, code: str) -> bool:
        """
        Detects low-level external calls whose boolean result is not checked.
        """
        for pattern in self.unchecked_external_call_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_start = code.rfind('\n', 0, match.start()) + 1
                line_end = code.find('\n', match.end())
                if line_end == -1:
                    line_end = len(code)
                line = code[line_start:line_end].strip()

                if re.search(r'\b(require|assert|if)\b', line, re.IGNORECASE):
                    continue
                if re.search(r'=\s*[^;]*\.(?:call|send|delegatecall|callcode)\s*\(', line, re.IGNORECASE):
                    continue
                if re.search(r'\(\s*bool\s+\w+', line, re.IGNORECASE):
                    continue
                return True
        return False
    
    def label_function(
        self,
        code: str,
        sol_version: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[str], Dict[str, bool]]:
        """
        Labels a function with all detected vulnerabilities.
        sol_version: (major, minor) from pragma; if >= (0, 8), skip integer overflow (native protection).
        """
        vulnerabilities = []
        detailed_labels = {}
        
        # Timestamp Dependency
        timestamp_labels = self.detect_timestamp_dependency(code)
        detailed_labels.update(timestamp_labels)
        if timestamp_labels['TimestampInvoc'] and (
            timestamp_labels['TimestampAssign'] or timestamp_labels['TimestampContaminate']
        ):
            vulnerabilities.append("Timestamp Dependency")
        
        # Reentrancy (skips if ReentrancyGuard/nonReentrant modifier present)
        if self.detect_reentrancy(code):
            vulnerabilities.append("Reentrancy")
            detailed_labels['Reentrancy'] = True
        
        # Integer Overflow/Underflow - skip for Solidity 0.8+ (native protection)
        skip_overflow = sol_version is not None and (sol_version[0] > 0 or sol_version[1] >= 8)
        if not skip_overflow and self.detect_integer_overflow(code):
            vulnerabilities.append("Integer Overflow/Underflow")
            detailed_labels['IntegerOverflow'] = True
        
        # Dangerous Delegatecall
        if self.detect_delegatecall(code):
            vulnerabilities.append("Dangerous Delegatecall")
            detailed_labels['Delegatecall'] = True

        # Transaction-Ordering Dependence
        if self.detect_transaction_ordering_dependence(code):
            vulnerabilities.append("Transaction-Ordering Dependence")
            detailed_labels['TransactionOrderingDependence'] = True

        # Uninitialized Storage Pointer
        if self.detect_uninitialized_storage_pointer(code):
            vulnerabilities.append("Uninitialized Storage Pointer")
            detailed_labels['UninitializedStoragePointer'] = True

        # Unchecked External Calls
        if self.detect_unchecked_external_calls(code):
            vulnerabilities.append("Unchecked External Calls")
            detailed_labels['UncheckedExternalCalls'] = True
        
        return vulnerabilities, detailed_labels


class SlitherExtractor:
    """Extracts function information using Slither"""
    
    def __init__(self, solc_version: Optional[str] = None, fallback_only: bool = False):
        self.solc_version = solc_version
        self.fallback_only = fallback_only
    
    def extract_functions(self, sol_file: str) -> List[Dict]:
        """
        Extracts functions from a Solidity file using Slither
        Returns list of function dictionaries
        """
        functions = []

        if self.fallback_only:
            return self._fallback_extract(sol_file)
        
        try:
            # Try to use Slither API
            from slither import Slither
            
            slither = Slither(sol_file)
            file_content = Path(sol_file).read_text(encoding="utf-8", errors="replace")
            file_lines = file_content.splitlines()
            
            for contract in slither.contracts:
                for function in contract.functions:
                    start_line = end_line = 0
                    code = ""
                    sm = function.source_mapping
                    if sm:
                        try:
                            # Slither may use dict or Source object (not subscriptable)
                            lines = sm.get("lines") if isinstance(sm, dict) else getattr(sm, "lines", None)
                            if lines:
                                start_line = lines[0] if lines else 0
                                end_line = lines[-1] if lines else 0
                            elif hasattr(sm, "start") and hasattr(sm, "length"):
                                # Byte offsets: convert to line range
                                start, length = sm.start, sm.length
                                upto = file_content[: start + length] if start is not None else ""
                                start_line = upto.count("\n") + 1 if upto else 0
                                end_line = file_content[: start + length].count("\n") + 1 if upto else 0
                            if start_line and end_line and file_lines:
                                code = "\n".join(file_lines[max(0, start_line - 1) : end_line])
                        except (TypeError, KeyError, IndexError, AttributeError):
                            pass
                    if not code and getattr(function, "nodes", None):
                        # Fallback: use first node's source
                        n = next(iter(function.nodes), None)
                        nsm = getattr(n, "source_mapping", None) if n else None
                        if nsm:
                            try:
                                lines = nsm.get("lines") if isinstance(nsm, dict) else getattr(nsm, "lines", None)
                                if lines:
                                    start_line, end_line = lines[0], lines[-1]
                                    code = "\n".join(file_lines[max(0, start_line - 1) : end_line])
                            except (TypeError, KeyError, IndexError, AttributeError):
                                pass
                    sig = getattr(function, "signature", "unknown(...)")
                    if isinstance(sig, list):
                        sig = sig[0] if sig else "unknown(...)"
                    func_data = {
                        "contract_name": getattr(contract, "name", "Unknown"),
                        "function_name": getattr(function, "name", "unknown"),
                        "function_signature": str(sig),
                        "start_line": start_line,
                        "end_line": end_line,
                        "visibility": str(getattr(function, "visibility", "public")),
                        "state_mutability": str(getattr(function, "state_mutability", "")),
                        "code": code or "",
                    }
                    functions.append(func_data)
            
            # If Slither returned no usable code, use fallback
            if functions and not any(f.get("code") for f in functions):
                functions = self._fallback_extract(sol_file)
        
        except ImportError:
            # Fallback: parse manually if Slither not available
            functions = self._fallback_extract(sol_file)
        
        except Exception as e:
            err_msg = str(e).lower()
            # Slither often fails: compilation, API changes, IR generation (unhashable type, etc.)
            if any(x in err_msg for x in (
                "invalid compilation", "compilation", "source", "subscriptable",
                "state_mutability", "unhashable", "parameters", "generate ir"
            )):
                functions = self._fallback_extract(sol_file)
            else:
                print(f"Error extracting from {sol_file}: {e}")
                functions = self._fallback_extract(sol_file)
        
        return functions
    
    def _fallback_extract(self, sol_file: str) -> List[Dict]:
        """Fallback function extraction using regex parsing"""
        functions = []
        
        try:
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract contract name
            contract_match = re.search(r'contract\s+(\w+)', content)
            contract_name = contract_match.group(1) if contract_match else "Unknown"
            
            # Extract functions
            # Pattern: function keyword, name, parameters, visibility, mutability
            func_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*(public|private|internal|external)?\s*(view|pure|payable)?\s*[^{]*\{'
            
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                visibility = match.group(2) or "public"
                mutability = match.group(3) or ""
                
                # Find function body
                start_pos = match.end()
                brace_count = 1
                end_pos = start_pos
                
                while end_pos < len(content) and brace_count > 0:
                    if content[end_pos] == '{':
                        brace_count += 1
                    elif content[end_pos] == '}':
                        brace_count -= 1
                    end_pos += 1
                
                func_code = content[match.start():end_pos]
                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:end_pos].count('\n') + 1
                
                functions.append({
                    'contract_name': contract_name,
                    'function_name': func_name,
                    'function_signature': f"{func_name}(...)",
                    'start_line': start_line,
                    'end_line': end_line,
                    'visibility': visibility,
                    'state_mutability': mutability,
                    'code': func_code
                })
            
            # Handle fallback function
            fallback_pattern = r'function\s*\(\)\s*(public|private|internal|external)?\s*(payable)?\s*[^{]*\{'
            for match in re.finditer(fallback_pattern, content):
                start_pos = match.end()
                brace_count = 1
                end_pos = start_pos
                
                while end_pos < len(content) and brace_count > 0:
                    if content[end_pos] == '{':
                        brace_count += 1
                    elif content[end_pos] == '}':
                        brace_count -= 1
                    end_pos += 1
                
                func_code = content[match.start():end_pos]
                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:end_pos].count('\n') + 1
                
                functions.append({
                    'contract_name': contract_name,
                    'function_name': "fallback",
                    'function_signature': "fallback()",
                    'start_line': start_line,
                    'end_line': end_line,
                    'visibility': match.group(1) or "public",
                    'state_mutability': match.group(2) or "",
                    'code': func_code
                })
        
        except Exception as e:
            print(f"Error in fallback extraction for {sol_file}: {e}")
        
        return functions


class DatasetStandardizer:
    """Main class for standardizing dataset formats"""
    
    def __init__(
        self,
        output_dir: str = "standardized_dataset",
        swc_mapping: Optional[Dict] = None,
        fallback_only: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.swc_mapping = swc_mapping or load_swc_mapping()
        self.extractor = SlitherExtractor(fallback_only=fallback_only)
        self.labeler = VulnerabilityLabeler()
        
        self.all_functions: List[FunctionData] = []
    
    def process_file(self, sol_file: str) -> List[FunctionData]:
        """Process a single .sol file"""
        functions_data = []
        sol_version = parse_solidity_version_from_file(sol_file)
        
        # Extract functions
        extracted_functions = self.extractor.extract_functions(sol_file)
        
        for func in extracted_functions:
            # Label vulnerabilities (pass sol_version to skip integer overflow for 0.8+)
            vulnerabilities, labels = self.labeler.label_function(func['code'], sol_version=sol_version)
            
            # Map to SWC IDs
            swc_ids = []
            for vuln in vulnerabilities:
                if vuln in self.swc_mapping:
                    swc_ids.append(self.swc_mapping[vuln]['swc_id'])
            
            # Create FunctionData object
            func_data = FunctionData(
                contract_file=sol_file,
                contract_name=func['contract_name'],
                function_name=func['function_name'],
                function_signature=func['function_signature'],
                function_code=func['code'],
                start_line=func['start_line'],
                end_line=func['end_line'],
                visibility=func['visibility'],
                state_mutability=func['state_mutability'],
                vulnerabilities=vulnerabilities,
                swc_ids=swc_ids,
                labels=labels,
                metadata={}
            )
            
            functions_data.append(func_data)
        
        return functions_data
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        validate: bool = True,
        skip_duplicates: bool = True,
        validation_min_length: int = 50,
    ):
        """
        Process all .sol files in a directory.
        Optionally filter invalid contracts (validate=True) and skip duplicate files (skip_duplicates=True).
        """
        directory = Path(directory)
        
        if recursive:
            sol_files = list(directory.rglob("*.sol"))
        else:
            sol_files = list(directory.glob("*.sol"))
        
        total_found = len(sol_files)
        print(f"Found {total_found} .sol files")
        
        # Apply validation and duplicate filtering if helpers available
        if filter_valid_solidity_files is not None and (validate or skip_duplicates):
            # Show progress every 1000 files when validating large sets
            progress_every = 1000 if (validate and total_found > 2000) else None
            if progress_every:
                print(f"Running full validation (progress every {progress_every} files)...")
            sol_files, filter_stats = filter_valid_solidity_files(
                sol_files,
                validate=validate,
                validation_min_length=validation_min_length,
                skip_duplicates=skip_duplicates,
                duplicate_normalize=True,
                use_canonical=True,
                progress_interval=progress_every,
            )
            print(f"After filtering: {filter_stats.valid} valid, {filter_stats.invalid} invalid, "
                  f"{filter_stats.duplicate_skipped} duplicates skipped ({filter_stats.duplicate_groups} groups)")
        else:
            sol_files = [Path(p) for p in sol_files]
        
        num_to_process = len(sol_files)
        progress_step = 1000 if num_to_process > 5000 else 100
        for i, sol_file in enumerate(sol_files, 1):
            if i % progress_step == 0:
                print(f"Processing file {i}/{num_to_process}...")
            
            try:
                functions = self.process_file(str(sol_file))
                self.all_functions.extend(functions)
            except Exception as e:
                print(f"Error processing {sol_file}: {e}")
                continue
    
    def export_json(self, filename: str = "standardized_dataset.json"):
        """Export to JSON format"""
        output_path = self.output_dir / filename
        
        data = {
            'metadata': {
                'total_functions': len(self.all_functions),
                'swc_mapping': self.swc_mapping,
                'format_version': '1.0'
            },
            'functions': [asdict(func) for func in self.all_functions]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(self.all_functions)} functions to {output_path}")
    
    def export_csv(self, filename: str = "standardized_dataset.csv"):
        """Export to CSV format"""
        output_path = self.output_dir / filename
        
        if not self.all_functions:
            print("No functions to export")
            return
        
        fieldnames = [
            'contract_file', 'contract_name', 'function_name', 'function_signature',
            'start_line', 'end_line', 'visibility', 'state_mutability',
            'vulnerabilities', 'swc_ids', 'has_reentrancy', 'has_timestamp_dependency',
            'has_integer_overflow', 'has_delegatecall', 'has_tod',
            'has_uninitialized_storage_pointer', 'has_unchecked_external_calls',
            'timestamp_invoc', 'timestamp_assign', 'timestamp_contaminate'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for func in self.all_functions:
                row = {
                    'contract_file': func.contract_file,
                    'contract_name': func.contract_name,
                    'function_name': func.function_name,
                    'function_signature': func.function_signature,
                    'start_line': func.start_line,
                    'end_line': func.end_line,
                    'visibility': func.visibility,
                    'state_mutability': func.state_mutability,
                    'vulnerabilities': ';'.join(func.vulnerabilities),
                    'swc_ids': ';'.join(func.swc_ids),
                    'has_reentrancy': 'Reentrancy' in func.vulnerabilities,
                    'has_timestamp_dependency': 'Timestamp Dependency' in func.vulnerabilities,
                    'has_integer_overflow': 'Integer Overflow/Underflow' in func.vulnerabilities,
                    'has_delegatecall': 'Dangerous Delegatecall' in func.vulnerabilities,
                    'has_tod': 'Transaction-Ordering Dependence' in func.vulnerabilities,
                    'has_uninitialized_storage_pointer': 'Uninitialized Storage Pointer' in func.vulnerabilities,
                    'has_unchecked_external_calls': 'Unchecked External Calls' in func.vulnerabilities,
                    'timestamp_invoc': func.labels.get('TimestampInvoc', False),
                    'timestamp_assign': func.labels.get('TimestampAssign', False),
                    'timestamp_contaminate': func.labels.get('TimestampContaminate', False)
                }
                writer.writerow(row)
        
        print(f"Exported {len(self.all_functions)} functions to {output_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Standardize Ethereum smart contract dataset formats'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing .sol files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='standardized_dataset',
        help='Output directory for standardized data'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format (json, csv, or both)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Process directories recursively'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip Solidity validation (process all .sol files)'
    )
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Do not skip duplicate files (by content hash)'
    )
    parser.add_argument(
        '--validation-min-length',
        type=int,
        default=50,
        help='Minimum file length in chars for validation (default: 50)'
    )
    parser.add_argument(
        '--fallback-only',
        action='store_true',
        help='Skip Slither and use regex-based function extraction only'
    )
    
    args = parser.parse_args()
    
    # Create standardizer
    standardizer = DatasetStandardizer(
        output_dir=args.output_dir,
        fallback_only=args.fallback_only,
    )
    
    # Process directory
    print(f"Processing directory: {args.input_dir}")
    standardizer.process_directory(
        args.input_dir,
        recursive=args.recursive,
        validate=not args.no_validate,
        skip_duplicates=not args.no_dedup,
        validation_min_length=args.validation_min_length,
    )
    
    # Export
    if args.format in ['json', 'both']:
        standardizer.export_json()
    
    if args.format in ['csv', 'both']:
        standardizer.export_csv()
    
    print(f"\nProcessing complete!")
    print(f"Total functions processed: {len(standardizer.all_functions)}")
    
    # Print statistics
    vuln_counts = defaultdict(int)
    for func in standardizer.all_functions:
        for vuln in func.vulnerabilities:
            vuln_counts[vuln] += 1
    
    print("\nVulnerability Statistics:")
    for vuln, count in sorted(vuln_counts.items()):
        print(f"  {vuln}: {count}")


if __name__ == "__main__":
    main()
