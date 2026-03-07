#!/usr/bin/env python3
"""
Helper functions for Ethereum smart contract datasets:
- Solidity validation (syntax, structure, optional compiler check)
- Duplicate detection (content hash, structural similarity)
"""

import os
import re
import hashlib
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Solidity validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a Solidity file or content."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid

    @property
    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append("Errors: " + "; ".join(self.errors))
        if self.warnings:
            parts.append("Warnings: " + "; ".join(self.warnings))
        return " | ".join(parts) if parts else "OK"


def _strip_solidity_comments(content: str) -> str:
    """Remove single-line (//) and multi-line (/* */) comments."""
    # Multi-line comments (non-greedy)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # Single-line comments
    content = re.sub(r'//[^\n]*', '', content)
    return content


def _normalize_whitespace(content: str) -> str:
    """Normalize whitespace: collapse spaces, strip lines, single newlines."""
    lines = (line.strip() for line in content.splitlines())
    return '\n'.join(line for line in lines if line)


def normalize_solidity_for_dedup(content: str) -> str:
    """
    Normalize Solidity source for duplicate detection:
    strip comments and normalize whitespace so formatting differences don't affect hash.
    """
    content = _strip_solidity_comments(content)
    content = _normalize_whitespace(content)
    return content


def validate_solidity_content(
    content: str,
    *,
    min_length: int = 50,
    require_pragma: bool = True,
    require_contract: bool = True,
    max_length: Optional[int] = None,
) -> ValidationResult:
    """
    Validate Solidity source content (no file I/O).
    Checks: non-empty, pragma, balanced braces, at least one 'contract', size limits.
    """
    result = ValidationResult(valid=True)

    if not content or not content.strip():
        result.valid = False
        result.errors.append("Empty or whitespace-only content")
        return result

    text = content.strip()
    length = len(text)

    if length < min_length:
        result.valid = False
        result.errors.append(f"Content too short (min {min_length} chars)")
        return result

    if max_length is not None and length > max_length:
        result.warnings.append(f"Content very long ({length} chars, max {max_length})")

    if require_pragma:
        if not re.search(r'\bpragma\s+solidity\s+', text, re.IGNORECASE):
            result.valid = False
            result.errors.append("Missing 'pragma solidity ...'")

    if require_contract:
        if not re.search(r'\bcontract\s+\w+', text, re.IGNORECASE):
            result.valid = False
            result.errors.append("No contract definition found")

    # Balanced braces (crude but fast)
    open_braces = text.count('{') - text.count('}')
    if open_braces != 0:
        result.valid = False
        result.errors.append(f"Unbalanced braces (difference: {open_braces})")

    # Check for unclosed string literals (odd number of unescaped quotes) – skip for very long files
    if length < 500_000:
        in_string = False
        i = 0
        while i < len(text):
            c = text[i]
            if c == '"' or c == "'":
                if i == 0 or text[i - 1] != '\\':
                    in_string = not in_string
            i += 1
        if in_string:
            result.warnings.append("Possible unclosed string literal")

    result.metadata["length"] = length
    return result


def validate_solidity_file(
    path: str | Path,
    *,
    min_length: int = 50,
    require_pragma: bool = True,
    require_contract: bool = True,
    max_length: Optional[int] = None,
    check_encoding: bool = True,
) -> ValidationResult:
    """
    Validate a .sol file: read content and run validate_solidity_content.
    Optionally checks that the file is readable and not binary.
    """
    path = Path(path)
    if not path.exists():
        return ValidationResult(valid=False, errors=[f"File not found: {path}"])
    if not path.is_file():
        return ValidationResult(valid=False, errors=[f"Not a file: {path}"])

    try:
        raw = path.read_bytes()
    except OSError as e:
        return ValidationResult(valid=False, errors=[f"Cannot read file: {e}"])

    if check_encoding:
        try:
            content = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return ValidationResult(valid=False, errors=["File is not valid UTF-8 (or binary)"])
    else:
        content = raw.decode("utf-8", errors="replace")

    result = validate_solidity_content(
        content,
        min_length=min_length,
        require_pragma=require_pragma,
        require_contract=require_contract,
        max_length=max_length,
    )
    result.metadata["path"] = str(path.resolve())
    result.metadata["size_bytes"] = len(raw)
    return result


def validate_solidity_with_solc(path: str | Path, solc_bin: Optional[str] = None) -> ValidationResult:
    """
    Validate by running solc (if available). Returns validation result;
    if solc is not found or fails, result.valid is False and errors describe the issue.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ValidationResult(valid=False, errors=[f"File not found or not a file: {path}"])

    bin_name = solc_bin or "solc"
    executable = shutil.which(bin_name)
    if not executable:
        return ValidationResult(
            valid=False,
            errors=[f"solc not found in PATH (tried: {bin_name}). Install Solidity compiler or set solc_bin."],
        )

    try:
        # Compile-only check; use temp dir for output (solc requires -o)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.mkdir(out_dir)
            proc = subprocess.run(
                [executable, "--bin", "-o", out_dir, str(path)],
                capture_output=True,
                timeout=60,
                text=True,
                cwd=str(path.parent),
            )
    except subprocess.TimeoutExpired:
        return ValidationResult(valid=False, errors=["solc timed out"])
    except Exception as e:
        return ValidationResult(valid=False, errors=[f"solc execution failed: {e}"])

    # Many projects use imports; solc may fail only due to missing deps. Treat as warning if we have contract-like content
    if proc.returncode != 0:
        content = path.read_text(encoding="utf-8", errors="replace")
        has_contract = bool(re.search(r'\bcontract\s+\w+', content, re.IGNORECASE))
        return ValidationResult(
            valid=False,
            errors=[f"solc exited with code {proc.returncode}", (proc.stderr or proc.stdout or "")[:500]],
            warnings=["Import or compiler errors (see errors). File may still be processable."] if has_contract else [],
        )
    return ValidationResult(valid=True, metadata={"solc": executable})


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def compute_content_hash(content: str, normalize: bool = True) -> str:
    """
    Compute a stable hash of Solidity source for duplicate detection.
    If normalize is True, comments are stripped and whitespace normalized.
    """
    if normalize:
        content = normalize_solidity_for_dedup(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_hash(path: str | Path, normalize: bool = True) -> Optional[str]:
    """Compute content hash of a file. Returns None if file cannot be read."""
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return compute_content_hash(content, normalize=normalize)
    except OSError:
        return None


def find_duplicate_files(
    file_paths: List[str | Path],
    *,
    normalize: bool = True,
) -> Dict[str, List[Path]]:
    """
    Group files by content hash. Returns a dict: hash -> list of paths.
    Only hashes with more than one file are included (actual duplicates).
    """
    path_list = [Path(p) for p in file_paths]
    hash_to_paths: Dict[str, List[Path]] = {}

    for p in path_list:
        if not p.is_file():
            continue
        h = compute_file_hash(p, normalize=normalize)
        if h is None:
            continue
        hash_to_paths.setdefault(h, []).append(p.resolve())

    return {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}


def get_duplicate_groups(
    file_paths: List[str | Path],
    *,
    normalize: bool = True,
) -> List[List[Path]]:
    """
    Return list of duplicate groups (each group is a list of paths with identical content).
    """
    dup_map = find_duplicate_files(file_paths, normalize=normalize)
    return list(dup_map.values())


def choose_canonical_from_group(paths: List[Path], prefer_short_path: bool = True) -> Path:
    """
    From a list of duplicate file paths, choose one as canonical (e.g. to keep, others to skip).
    If prefer_short_path, chooses the path with the fewest path components; otherwise first in list.
    """
    if not paths:
        raise ValueError("paths must be non-empty")
    if prefer_short_path:
        return min(paths, key=lambda p: len(p.parts))
    return paths[0]


def extract_contract_signature(content: str) -> str:
    """
    Extract a minimal structural signature: contract names and function names (no bodies).
    Used for structural duplicate detection (same contract layout, possibly different formatting).
    """
    content = _strip_solidity_comments(content)
    parts = []

    # Contract names
    for m in re.finditer(r'\bcontract\s+(\w+)', content, re.IGNORECASE):
        parts.append(f"contract:{m.group(1)}")

    # Function names (including fallback)
    for m in re.finditer(r'\bfunction\s+(\w*)\s*\(', content, re.IGNORECASE):
        name = m.group(1) or "fallback"
        parts.append(f"fn:{name}")

    return "\n".join(parts)


def compute_structural_hash(content: str) -> str:
    """Hash based on contract and function names only (structural duplicate detection)."""
    sig = extract_contract_signature(content)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def find_structural_duplicates(
    file_paths: List[str | Path],
) -> Dict[str, List[Path]]:
    """
    Group files by structural hash (contract + function names). Returns hash -> list of paths.
    Only groups with more than one file are returned.
    """
    path_list = [Path(p) for p in file_paths]
    hash_to_paths: Dict[str, List[Path]] = {}

    for p in path_list:
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        h = compute_structural_hash(content)
        hash_to_paths.setdefault(h, []).append(p.resolve())

    return {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}


# ---------------------------------------------------------------------------
# Filtered file listing (for pipeline integration)
# ---------------------------------------------------------------------------

@dataclass
class FilterStats:
    """Statistics from filtering a list of files."""
    total: int = 0
    valid: int = 0
    invalid: int = 0
    duplicate_skipped: int = 0
    duplicate_groups: int = 0
    invalid_paths: List[Path] = field(default_factory=list)
    duplicate_groups_list: List[List[Path]] = field(default_factory=list)


def filter_valid_solidity_files(
    file_paths: List[str | Path],
    *,
    validate: bool = True,
    validation_min_length: int = 50,
    skip_duplicates: bool = True,
    duplicate_normalize: bool = True,
    use_canonical: bool = True,
    progress_interval: Optional[int] = None,
) -> Tuple[List[Path], FilterStats]:
    """
    From a list of .sol paths, return (valid_paths, stats).
    - If validate is True, only files that pass validate_solidity_file are kept.
    - If skip_duplicates is True, from each content-duplicate group only one (canonical) path is kept.
    - use_canonical: when skipping duplicates, keep the canonical (shortest path) per group.
    - progress_interval: if set, print progress every N files during validation (e.g. 1000).
    """
    paths = [Path(p).resolve() for p in file_paths]
    stats = FilterStats(total=len(paths))

    if not paths:
        return [], stats

    # 1) Validation
    if validate:
        valid_paths = []
        total = len(paths)
        for i, p in enumerate(paths):
            if progress_interval and (i + 1) % progress_interval == 0:
                print(f"  Validating {i + 1}/{total}... (valid so far: {len(valid_paths)})")
            if not p.is_file():
                stats.invalid += 1
                stats.invalid_paths.append(p)
                continue
            res = validate_solidity_file(p, min_length=validation_min_length)
            if res.valid:
                valid_paths.append(p)
            else:
                stats.invalid += 1
                stats.invalid_paths.append(p)
        paths = valid_paths

    stats.valid = len(paths)

    # 2) Duplicate handling
    if skip_duplicates and paths:
        dup_map = find_duplicate_files(paths, normalize=duplicate_normalize)
        stats.duplicate_groups_list = list(dup_map.values())
        stats.duplicate_groups = len(dup_map)

        all_duplicate_paths: Set[Path] = set()
        canonical_paths: Set[Path] = set()
        for group in dup_map.values():
            for p in group:
                all_duplicate_paths.add(p)
            canonical = choose_canonical_from_group(group, prefer_short_path=use_canonical)
            canonical_paths.add(canonical)

        # From paths, keep: non-duplicates + canonical of each duplicate group
        kept = []
        for p in paths:
            if p not in all_duplicate_paths:
                kept.append(p)
            elif p in canonical_paths:
                kept.append(p)
            else:
                stats.duplicate_skipped += 1
        paths = kept

    return paths, stats
