# Ethereum Smart Contract Dataset Standardization

This tool standardizes Ethereum smart contract datasets by:
1. Extracting functions from `.sol` files using Slither
2. Applying exact labeling rules for vulnerability detection
3. Mapping labels to SWC IDs
4. Outputting standardized JSON/CSV formats

## Vulnerability Types and SWC Mapping

| Vulnerability | SWC ID | SWC Name |
|--------------|--------|----------|
| Reentrancy | SWC-107 | Reentrancy |
| Timestamp Dependency | SWC-116 | Block Timestamp Dependence |
| Integer Overflow/Underflow | SWC-101 | Integer Overflow and Underflow |
| Dangerous Delegatecall | SWC-112 | Delegatecall to Untrusted Contract |

## Labeling Rules

### Timestamp Dependency (SWC-116)
Rule: `TimestampInvoc ∧ (TimestampAssign ∨ TimestampContaminate)`

- **TimestampInvoc**: Function invokes timestamp-related variables (`now`, `block.timestamp`, `block.number`)
- **TimestampAssign**: Function assigns timestamp values to variables
- **TimestampContaminate**: Timestamp is used in conditional operations or logic

### Reentrancy (SWC-107)
Detects external calls (`.call()`, `.send()`, `.transfer()`, `.delegatecall()`) that may allow reentrancy attacks.

### Integer Overflow/Underflow (SWC-101)
Detects arithmetic operations that may cause integer overflow or underflow.

### Dangerous Delegatecall (SWC-112)
Detects usage of `delegatecall` which can be dangerous if used with untrusted contracts.

## Mythril Supplement (Research Dataset)

To supplement the primary dataset with Mythril scans (as in your research methodology):

1. **Install Mythril:** `py -m pip install mythril` (requires solc on PATH)
2. **Copy and scan contracts:**
   ```cmd
   py mythril_scan.py contract_dataset_ethereum --output-dir mythril_scan_output --copy-staging
   ```
3. **Merge with primary dataset:**
   ```cmd
   py merge_mythril_dataset.py --primary standardized_dataset/standardized_dataset.json --mythril mythril_scan_output/mythril_scan_results.json -o combined_dataset.json
   ```

Mythril targets: Reentrancy, Integer overflow/underflow, Transaction-ordering dependence (front-running), Uninitialized storage pointers, Unchecked external calls. See `mythril_swc_mapping.json` for SWC mappings.

## Installation

```bash
pip install -r requirements.txt
```

Note: Slither requires `solc` (Solidity compiler) to be installed. See [Slither documentation](https://github.com/crytic/slither) for installation instructions.

## Usage

### Basic Usage

```bash
python standardize_dataset.py <input_directory> --output-dir standardized_dataset --format both
```

### Arguments

- `input_dir`: Directory containing `.sol` files (required)
- `--output-dir`: Output directory for standardized data (default: `standardized_dataset`)
- `--format`: Output format - `json`, `csv`, or `both` (default: `both`)
- `--recursive`: Process directories recursively (default: True)
- `--no-validate`: Skip Solidity validation (process all .sol files)
- `--no-dedup`: Do not skip duplicate files (by content hash)
- `--validation-min-length`: Minimum file length in characters for validation (default: 50)

### Examples

```bash
# Process all contracts in contract_dataset_ethereum
python standardize_dataset.py Ethereum_smart_contract_datast/contract_dataset_ethereum

# Export only JSON format
python standardize_dataset.py Ethereum_smart_contract_datast/contract_dataset_ethereum --format json

# Process specific directory
python standardize_dataset.py Ethereum_smart_contract_datast/contract_dataset_ethereum/contract1
```

## Output Format

### JSON Format (`standardized_dataset.json`)

```json
{
  "metadata": {
    "total_functions": 1000,
    "swc_mapping": {...},
    "format_version": "1.0"
  },
  "functions": [
    {
      "contract_file": "path/to/contract.sol",
      "contract_name": "MyContract",
      "function_name": "vulnerableFunction",
      "function_signature": "vulnerableFunction(uint256)",
      "function_code": "function vulnerableFunction(uint256 x) public {...}",
      "start_line": 10,
      "end_line": 25,
      "visibility": "public",
      "state_mutability": "",
      "vulnerabilities": ["Timestamp Dependency"],
      "swc_ids": ["SWC-116"],
      "labels": {
        "TimestampInvoc": true,
        "TimestampAssign": false,
        "TimestampContaminate": true
      },
      "metadata": {}
    }
  ]
}
```

### CSV Format (`standardized_dataset.csv`)

The CSV includes columns:
- `contract_file`, `contract_name`, `function_name`, `function_signature`
- `start_line`, `end_line`, `visibility`, `state_mutability`
- `vulnerabilities` (semicolon-separated)
- `swc_ids` (semicolon-separated)
- `has_reentrancy`, `has_timestamp_dependency`, `has_integer_overflow`, `has_delegatecall`
- `timestamp_invoc`, `timestamp_assign`, `timestamp_contaminate`

## Function Extraction

The tool uses Slither for accurate function extraction. If Slither is not available, it falls back to a regex-based parser (less accurate but functional).

### Slither Extraction (Recommended)
- Accurate function boundaries
- Proper signature extraction
- Visibility and mutability detection

### Fallback Extraction
- Regex-based parsing
- Basic function detection
- May miss some edge cases

## Labeling Logic

The labeling system applies pattern matching and heuristics to detect vulnerabilities:

1. **Timestamp Dependency**: Checks for timestamp invocations and their usage patterns
2. **Reentrancy**: Identifies external calls that may be vulnerable
3. **Integer Overflow**: Detects arithmetic operations on integers
4. **Delegatecall**: Finds delegatecall usage

Note: These are heuristic-based detections. For production use, consider integrating with more sophisticated static analysis tools.

## Helper Functions (`helpers.py`)

The `helpers` module provides **Solidity validation** and **duplicate detection** to filter invalid contracts and detect duplicates before or during processing.

### Solidity validation

- **`validate_solidity_content(content, ...)`** — Validate source string: non-empty, `pragma solidity`, at least one `contract`, balanced braces, optional length limits.
- **`validate_solidity_file(path, ...)`** — Validate a `.sol` file (read + `validate_solidity_content`). Options: `min_length`, `require_pragma`, `require_contract`, `max_length`, `check_encoding`.
- **`validate_solidity_with_solc(path, solc_bin=None)`** — Validate by running the Solidity compiler (if available).

### Duplicate detection

- **`normalize_solidity_for_dedup(content)`** — Strip comments and normalize whitespace for stable hashing.
- **`compute_content_hash(content, normalize=True)`** / **`compute_file_hash(path, normalize=True)`** — SHA-256 hash of (optionally normalized) content.
- **`find_duplicate_files(file_paths, normalize=True)`** — Returns `{ hash -> list of paths }` for groups with more than one file (content-based duplicates).
- **`get_duplicate_groups(file_paths, ...)`** — Returns `List[List[Path]]` of duplicate groups.
- **`choose_canonical_from_group(paths, prefer_short_path=True)`** — Pick one path per group (e.g. to keep, others to skip).
- **`extract_contract_signature(content)`** / **`compute_structural_hash(content)`** — Structural signature (contract + function names only).
- **`find_structural_duplicates(file_paths)`** — Group files by structural hash (same layout, possibly different formatting).

### Filtered file listing

- **`filter_valid_solidity_files(file_paths, validate=True, skip_duplicates=True, ...)`** — Returns `(list of paths to process, FilterStats)`. Use this to get a cleaned list before processing.

The standardizer uses these helpers when you run `process_directory()`: by default it validates each file and skips content-duplicate copies (keeping one canonical path per group). Disable with `--no-validate` or `--no-dedup`.

## Testing that labeling works

To check that vulnerability labeling runs successfully:

```bash
# Run unit + integration tests (snippet-based + one real .sol file if present)
python test_labeling.py
```

If all tests pass, you’ll see: **All tests passed — labeling is working successfully.**

To inspect labels for a specific contract without running the full pipeline:

```bash
# Print per-function vulnerabilities and SWC IDs for one file
python test_labeling.py --report path/to/contract.sol
```

The test file `test_labeling.py` checks that:

- **Timestamp**: code using `now` / `block.timestamp` gets `TimestampInvoc`; in conditions it gets `TimestampContaminate` and the full *Timestamp Dependency* (SWC-116).
- **Reentrancy**: code with `.transfer()` / `.call()` gets *Reentrancy* (SWC-107).
- **Integer overflow**: code with `++` / `+=` gets *Integer Overflow/Underflow* (SWC-101).
- **Delegatecall**: code with `.delegatecall()` gets *Dangerous Delegatecall* (SWC-112).
- **SWC mapping**: every vulnerability type has a valid SWC ID.
- **Integration**: processing a real `.sol` file returns `FunctionData` with `vulnerabilities`, `swc_ids`, and `labels` populated.

## Count valid contracts and vulnerabilities

To see **how many valid contracts** you have and **how many have each vulnerability** (and how many have all four):

**From a directory of `.sol` files** (validates and deduplicates, then counts):

```bash
py report_vulnerability_counts.py contract_dataset_ethereum
```

**From an existing standardized JSON** (no rescan):

```bash
py report_vulnerability_counts.py --from-json standardized_dataset/standardized_dataset.json
```

**Options when using a directory:**

- `--no-validate` — do not filter invalid contracts
- `--no-dedup` — do not skip duplicate files

**Example output:**

```
============================================================
Vulnerability counts (valid contracts)
============================================================
  Total valid contracts:  150
  Total functions:       1823

  Per vulnerability (contracts with ≥1 function | function count):
  --------------------------------------------------
    Reentrancy                     contracts:    42  functions:   89  (SWC-107)
    Timestamp Dependency          contracts:    67  functions:  112  (SWC-116)
    Integer Overflow/Underflow    contracts:   101  functions:  456  (SWC-101)
    Dangerous Delegatecall        contracts:    12  functions:   14  (SWC-112)
  --------------------------------------------------
  Contracts with ALL 4 vulnerabilities:  3
============================================================
```

So you get: total valid contracts, total functions, per-vulnerability contract/function counts, and how many contracts have all four vulnerability types.

## Running full validation

Yes. By default, both **standardize_dataset.py** and **report_vulnerability_counts.py** use **full validation** (and duplicate filtering). You do not need to pass any flag.

- **Standardize (with validation):**
  ```cmd
  py standardize_dataset.py contract_dataset_ethereum
  ```
- **Vulnerability counts (with validation):**
  ```cmd
  py report_vulnerability_counts.py contract_dataset_ethereum
  ```

For **large datasets** (e.g. tens of thousands of files), the run can take a long time. Progress is printed so you can see it’s still running:

- **Validation:** every 1000 files you’ll see e.g. `Validating 1000/40749... (valid so far: 892)`.
- **Processing:** every 1000 files (or every 100 for smaller sets) you’ll see e.g. `Processing file 5000/35000...`.

Let it run to completion; do not press Ctrl+C if you want full results. To skip validation and run faster, use `--no-validate` (and optionally `--no-dedup`).

## Extending the Tool

To add new vulnerability types:

1. Add SWC mapping in `SWC_MAPPING` dictionary
2. Implement detection logic in `VulnerabilityLabeler` class
3. Update `label_function` method to include new detection
4. Update CSV export fieldnames if needed

## Troubleshooting

### "Python was not found" (Windows)

On Windows, the Python launcher is often `py` instead of `python`. Try in order:

1. **`py`** (Windows Python launcher):
   ```cmd
   py test_labeling.py --report contract_dataset_ethereum\contract1\0.sol
   ```
2. **`python3`**:
   ```cmd
   python3 test_labeling.py --report contract_dataset_ethereum\contract1\0.sol
   ```
3. **`python`** (only if one of the above is not installed):
   ```cmd
   python test_labeling.py --report contract_dataset_ethereum\contract1\0.sol
   ```

Or use the batch scripts (they try `py`, then `python3`, then `python`):

- **Inspect labels for one contract:** double‑click or run:
  ```cmd
  run_report.bat contract_dataset_ethereum\contract1\0.sol
  ```
- **Run all labeling tests:**
  ```cmd
  run_tests.bat
  ```

If none of these work, [install Python](https://www.python.org/downloads/) and during setup check **"Add Python to PATH"**, then open a new terminal.

### "No such file or directory" for test_labeling.py

This usually means the command was run from the **wrong folder**. The script lives in the folder that also contains `standardize_dataset.py`, `helpers.py`, etc.

1. **Open a terminal and go to that folder first**, then run the script:
   ```cmd
   cd C:\Users\HUAWEI\Downloads\Ethereum_smart_contract_datast\Ethereum_smart_contract_datast
   py test_labeling.py --report contract_dataset_ethereum\contract1\0.sol
   ```
   (If your project has only one `Ethereum_smart_contract_datast` folder, use:
   `cd C:\Users\HUAWEI\Downloads\Ethereum_smart_contract_datast` instead.)

2. **Or call the script by full path** (no need to cd):
   ```cmd
   py "C:\Users\HUAWEI\Downloads\Ethereum_smart_contract_datast\Ethereum_smart_contract_datast\test_labeling.py" --report "C:\Users\HUAWEI\Downloads\Ethereum_smart_contract_datast\Ethereum_smart_contract_datast\contract_dataset_ethereum\contract1\0.sol"
   ```
   Adjust paths if your project is under a different folder.

3. **Or use the batch file from the project root**: if there is a `run_report.bat` in the parent folder, run:
   ```cmd
   run_report.bat Ethereum_smart_contract_datast\contract_dataset_ethereum\contract1\0.sol
   ```

### Slither Installation Issues
If Slither fails to install, ensure:
- Python 3.8+ is installed
- `solc` (Solidity compiler) is installed and in PATH
- See [Slither installation guide](https://github.com/crytic/slither#how-to-install)

### Memory Issues
For large datasets:
- Process directories in batches
- Use `--format json` or `--format csv` instead of `both` to reduce memory usage

### Parsing Errors
Some contracts may fail to parse due to:
- Syntax errors in source files
- Unsupported Solidity versions
- Missing dependencies

The tool will skip problematic files and continue processing.

## License

This tool is provided as-is for dataset standardization purposes.
