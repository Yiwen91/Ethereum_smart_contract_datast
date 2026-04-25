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
| Transaction-Ordering Dependence | SWC-114 | Transaction-Ordering Dependence |
| Uninitialized Storage Pointer | SWC-109 | Uninitialized Storage Pointer |
| Unchecked External Calls | SWC-104 | Unchecked Call Return Value |

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

### Transaction-Ordering Dependence (SWC-114)
Detects public or external functions that update shared order-sensitive state and use competitive logic or value flow that may be front-run.

### Uninitialized Storage Pointer (SWC-109)
Detects local storage variables declared without initialization, which can alias unintended storage slots in older Solidity versions.

### Unchecked External Calls (SWC-104)
Detects low-level external calls such as `.call()` and `.send()` when their return values are ignored.

## Installation

```bash
pip install -r requirements.txt
```

Note: Slither requires `solc` (Solidity compiler) to be installed. See [Slither documentation](https://github.com/crytic/slither) for installation instructions.
If Slither is unavailable or fails on a contract, the pipeline can fall back to regex-based function extraction.

## Usage

### Basic Usage

```bash
python standardize_dataset.py <input_directory> --output-dir standardized_dataset --format both
```

## Dataset Workflows

This project now supports two dataset roles:

- **Primary dataset**: Messi-Q-derived processed Ethereum dataset stored in `contract_dataset_ethereum` (ESC)
- **Secondary dataset**: SmartBugs Wild raw contracts stored in `smartbugs_wild/contracts`

### Quick Comparison

| Dataset | Folder | Role | Best helper |
|--------------|--------|------|-------------|
| Dataset 1: Ethereum smart contract | `contract_dataset_ethereum` | Primary processed dataset | `run_dataset1_ethereum.bat` |
| Dataset 2: SmartBugs Wild | `smartbugs_wild/contracts` | Secondary raw dataset | `run_dataset2_smartbugs.bat` |

### Primary Dataset: Ethereum Smart Contracts (ESC)

The primary dataset is:

- `contract_dataset_ethereum`

Run the main ESC pipeline:

```cmd
py standardize_dataset.py contract_dataset_ethereum --output-dir standardized_dataset --format both
py report_vulnerability_counts.py contract_dataset_ethereum -o vulnerability_report.txt
```

Or use the dataset-specific helper:

```cmd
run_dataset1_ethereum.bat
```

Inspect one ESC contract:

```cmd
inspect_dataset1_contract.bat contract_dataset_ethereum\contract1\0.sol
```

Create balanced experiment splits from the labeled ESC JSON:

```cmd
py prepare_experiment_splits.py --from-json standardized_dataset\standardized_dataset.json --output-dir experiment_splits\esc_primary
```

This creates contract-level `train`, `val`, and `test` splits while balancing the 7 configured vulnerability types as evenly as possible across the splits.

Train the first ESC tabular baseline on those split files:

```cmd
py train_experiment.py --model tabular --split-dir experiment_splits\esc_primary --output-dir experiments\tabular_baseline
```

The baseline uses:

- function-level multilabel targets
- TF-IDF features from `function_code`
- one-vs-rest logistic regression
- validation-tuned per-label thresholds

By default, the training script uses sample limits for a faster first run:

- `--max-train-samples 100000`
- `--max-val-samples 20000`
- `--max-test-samples 20000`

Outputs are saved under the chosen experiment folder and include:

- `run_config.json`
- `thresholds.json`
- `val_metrics.json`
- `test_metrics.json`
- `summary.txt`
- `val_predictions.jsonl`
- `test_predictions.jsonl`

The experiment reports now include:

- micro precision / recall / F1
- macro precision / recall / F1
- weighted precision / recall / F1
- micro / macro / weighted AUC-ROC
- subset accuracy
- inference latency
- per-label accuracy / precision / recall / F1 / AUC-ROC

Run multiple seeds and aggregate mean/std for stronger academic reporting:

```cmd
py train_experiment.py --model tabular --split-dir experiment_splits\esc_primary --output-dir experiments\tabular_baseline --run-name esc_tabular_multiseed --max-train-samples 50000 --max-val-samples 10000 --max-test-samples 10000 --sample-strategy reservoir --seeds 42 43 44
```

This writes per-seed run folders plus an aggregate folder containing:

- `aggregate_metrics.json`
- `aggregate_summary.txt`

Train a `CodeBERT` semantic baseline on the same ESC split files:

```cmd
py train_experiment.py --model codebert --split-dir experiment_splits\esc_primary --output-dir experiments\codebert_baseline --run-name codebert_smoke_test --max-train-samples 1000 --max-val-samples 200 --max-test-samples 200 --sample-strategy head --epochs 1 --train-batch-size 4 --eval-batch-size 8 --max-length 256
```

The `CodeBERT` baseline uses:

- the same function-level multilabel targets
- the same train / val / test split files
- the same threshold tuning and metrics as the tabular baseline
- `microsoft/codebert-base` as the default pretrained model

Useful `CodeBERT` options:

- `--codebert-model-name` to switch pretrained checkpoints
- `--epochs` to control fine-tuning length
- `--train-batch-size` and `--eval-batch-size` for memory control
- `--max-length` to cap tokenized function length
- `--device cpu` or `--device cuda` to force device choice
- `--save-model` to store the fine-tuned model and tokenizer in the run folder
- `--max-pos-weight` to cap rare-label positive weighting during BCE training
- `--grad-clip-norm` to stabilize transformer fine-tuning
- `--default-threshold`, `--threshold-min-support`, and `--threshold-min-precision` to keep threshold tuning from collapsing to overly low values on rare labels

A safer larger `CodeBERT` command for Colab GPU is:

```cmd
python train_experiment.py --model codebert --split-dir experiment_splits/esc_primary --output-dir experiments/codebert_baseline --run-name esc_codebert_tuned --max-train-samples 10000 --max-val-samples 1500 --max-test-samples 1500 --sample-strategy reservoir --epochs 3 --train-batch-size 8 --eval-batch-size 8 --max-length 128 --learning-rate 2e-5 --max-pos-weight 8 --grad-clip-norm 1.0 --default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.15
```

Run multiple tuned `CodeBERT` seeds and report mean/std:

```cmd
python train_experiment.py --model codebert --codebert-model-name hf_models/hf_models/codebert-base --split-dir experiment_splits/esc_primary --output-dir experiments/codebert_baseline --run-name esc_codebert_multiseed --max-train-samples 10000 --max-val-samples 1500 --max-test-samples 1500 --sample-strategy reservoir --epochs 3 --train-batch-size 8 --eval-batch-size 8 --max-length 128 --learning-rate 2e-5 --max-pos-weight 8 --grad-clip-norm 1.0 --default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.15 --seeds 42 43 44
```

Train an `AST/CFG GNN` structural baseline on the same ESC split files:

```cmd
py train_experiment.py --model gnn --split-dir experiment_splits\esc_primary --output-dir experiments\gnn_baseline --run-name gnn_smoke_test --max-train-samples 5000 --max-val-samples 1000 --max-test-samples 1000 --sample-strategy reservoir --gnn-epochs 3 --gnn-max-nodes 48 --gnn-feature-dim 256 --gnn-hidden-dim 128 --gnn-num-layers 2 --gnn-train-batch-size 64 --gnn-eval-batch-size 128
```

The `GNN` baseline uses:

- the same function-level multilabel targets
- the same train / val / test split files
- Solidity AST nodes extracted from `solc`
- CFG nodes extracted from `Slither`
- a fused AST/CFG function graph with cross-edges by source-line overlap
- hashed node features extracted from node types and code snippets
- the same threshold tuning and saved metrics as the other baselines

Useful `GNN` options:

- `--gnn-max-nodes` to cap fused AST/CFG graph size per function
- `--gnn-feature-dim` to control hashed node feature width
- `--gnn-hidden-dim` and `--gnn-num-layers` to scale model capacity
- `--gnn-train-batch-size` and `--gnn-eval-batch-size` for memory control
- `--gnn-learning-rate`, `--gnn-weight-decay`, and `--gnn-epochs` for tuning
- `--gnn-max-pos-weight` and `--gnn-grad-clip-norm` for imbalance and optimization stability

A safer larger `AST/CFG GNN` command for Colab is:

```cmd
python train_experiment.py --model gnn --split-dir experiment_splits/esc_primary --output-dir experiments/gnn_baseline --run-name esc_gnn_tuned_100k --max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 --sample-strategy reservoir --gnn-epochs 3 --gnn-max-nodes 48 --gnn-feature-dim 256 --gnn-hidden-dim 128 --gnn-num-layers 2 --gnn-dropout 0.2 --gnn-train-batch-size 64 --gnn-eval-batch-size 128 --gnn-learning-rate 1e-3 --gnn-weight-decay 1e-4 --gnn-max-pos-weight 8 --gnn-grad-clip-norm 1.0 --default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.15
```

Train a `Hybrid CodeBERT + AST/CFG GNN` model on the same ESC split files:

```cmd
py train_experiment.py --model hybrid --split-dir experiment_splits\esc_primary --output-dir experiments\hybrid_baseline --run-name hybrid_smoke_test --max-train-samples 2000 --max-val-samples 400 --max-test-samples 400 --sample-strategy reservoir --hybrid-epochs 1 --hybrid-train-batch-size 2 --hybrid-eval-batch-size 4 --max-length 192 --hybrid-max-nodes 96 --hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 --hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.2 --hybrid-gradient-accumulation-steps 2 --hybrid-checkpoint-metric micro_f1 --threshold-candidates 0.4 0.5 0.6 0.7 0.8
```

The `Hybrid` model uses:

- `CodeBERT` as the semantic encoder over raw function code
- the fused `AST/CFG` graph builder from the structural baseline
- text-anchored modality attention plus a gated graph residual before multilabel prediction
- the same threshold tuning, saved metrics, and per-label reporting as the other baselines

Useful `Hybrid` options:

- `--hybrid-train-batch-size` and `--hybrid-eval-batch-size` for GPU memory control
- `--hybrid-transformer-learning-rate` and `--hybrid-head-learning-rate` to tune the encoder and fusion head separately
- `--hybrid-fusion-dim`, `--hybrid-attention-heads`, `--hybrid-graph-residual-scale`, and `--hybrid-dropout` for the fusion block
- `--hybrid-max-nodes`, `--hybrid-feature-dim`, `--hybrid-graph-hidden-dim`, and `--hybrid-graph-num-layers` for the graph branch
- `--hybrid-max-pos-weight`, `--hybrid-grad-clip-norm`, `--hybrid-gradient-accumulation-steps`, and `--hybrid-encoder-warmup-epochs` for stability
- `--hybrid-checkpoint-metric` to save the best epoch by validation `micro_f1`, `weighted_f1`, or `subset_accuracy`
- `--threshold-candidates` to search a finer threshold grid when optimizing headline F1

A best-shot headline `Hybrid` command for Colab is:

```cmd
python train_experiment.py --model hybrid --codebert-model-name microsoft/codebert-base --split-dir experiment_splits/esc_primary --output-dir experiments/hybrid_baseline --run-name esc_hybrid_headline_100k --max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 --sample-strategy reservoir --hybrid-epochs 4 --hybrid-train-batch-size 2 --hybrid-eval-batch-size 4 --max-length 192 --hybrid-max-nodes 96 --hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 --hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.2 --hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1.5e-5 --hybrid-head-learning-rate 7e-4 --hybrid-weight-decay 0.01 --hybrid-max-pos-weight 8 --hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric micro_f1 --default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.0 --threshold-candidates 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9
```

If the best-shot run does not improve enough, try this fallback with slightly weaker graph influence and a slightly lower head LR:

```cmd
python train_experiment.py --model hybrid --codebert-model-name microsoft/codebert-base --split-dir experiment_splits/esc_primary --output-dir experiments/hybrid_baseline --run-name esc_hybrid_headline_fallback_100k --max-train-samples 100000 --max-val-samples 10000 --max-test-samples 10000 --sample-strategy reservoir --hybrid-epochs 4 --hybrid-train-batch-size 2 --hybrid-eval-batch-size 4 --max-length 192 --hybrid-max-nodes 96 --hybrid-feature-dim 256 --hybrid-graph-hidden-dim 128 --hybrid-graph-num-layers 2 --hybrid-fusion-dim 256 --hybrid-attention-heads 4 --hybrid-graph-residual-scale 0.12 --hybrid-dropout 0.15 --hybrid-transformer-learning-rate 1.5e-5 --hybrid-head-learning-rate 5e-4 --hybrid-weight-decay 0.01 --hybrid-max-pos-weight 8 --hybrid-grad-clip-norm 1.0 --hybrid-gradient-accumulation-steps 4 --hybrid-encoder-warmup-epochs 1 --hybrid-checkpoint-metric micro_f1 --default-threshold 0.5 --threshold-min-support 5 --threshold-min-precision 0.0 --threshold-candidates 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9
```

### Secondary Dataset: SmartBugs Wild

Import the raw SmartBugs Wild contracts into this repo:

```cmd
py import_smartbugs_wild.py --include-metadata
```

This places the secondary dataset under:

- `smartbugs_wild/contracts`

Run the full secondary-dataset pipeline:

```cmd
py standardize_dataset.py smartbugs_wild\contracts --output-dir standardized_smartbugs --format both --fallback-only --no-validate --no-dedup
py report_vulnerability_counts.py --from-json standardized_smartbugs\standardized_dataset.json -o smartbugs_vulnerability_report.txt
```

Or use the helper batch file:

```cmd
run_dataset2_smartbugs.bat
```

Inspect one SmartBugs contract:

```cmd
inspect_dataset2_smartbugs_contract.bat smartbugs_wild\contracts\<contract>.sol
```

`smartbugs-wild` is already published as a curated benchmark dataset with duplicates removed, so the secondary workflow skips validation and dedup for the full run to reduce runtime while preserving the source dataset layout.

### Side-By-Side Commands

Run full dataset 1:

```cmd
run_dataset1_ethereum.bat
```

Run full dataset 2:

```cmd
run_dataset2_smartbugs.bat
```

Inspect one dataset 1 contract:

```cmd
inspect_dataset1_contract.bat contract_dataset_ethereum\contract1\0.sol
```

Inspect one dataset 2 contract:

```cmd
inspect_dataset2_smartbugs_contract.bat smartbugs_wild\contracts\<contract>.sol
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

# Process SmartBugs Wild as the secondary dataset
python standardize_dataset.py Ethereum_smart_contract_datast/smartbugs_wild/contracts --output-dir standardized_smartbugs --format both --fallback-only --no-validate --no-dedup

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
- `has_tod`, `has_uninitialized_storage_pointer`, `has_unchecked_external_calls`
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
5. **Transaction-Ordering Dependence**: Looks for shared order-sensitive state updates in public/external functions
6. **Uninitialized Storage Pointer**: Finds local storage declarations without initialization
7. **Unchecked External Calls**: Flags low-level calls whose boolean results are ignored

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
- **Transaction ordering dependence**: code with bidding/order-sensitive shared state gets *Transaction-Ordering Dependence* (SWC-114).
- **Storage pointers**: uninitialized local `storage` declarations get *Uninitialized Storage Pointer* (SWC-109).
- **Unchecked external calls**: ignored low-level `.call()` / `.send()` results get *Unchecked External Calls* (SWC-104).
- **SWC mapping**: every vulnerability type has a valid SWC ID.
- **Integration**: processing a real `.sol` file returns `FunctionData` with `vulnerabilities`, `swc_ids`, and `labels` populated.

## Count valid contracts and vulnerabilities

To see **how many valid contracts** you have and **how many have each vulnerability** (and how many have all configured vulnerability types):

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
    Transaction-Ordering Dependence contracts:   23  functions:   41  (SWC-114)
    Uninitialized Storage Pointer contracts:     8  functions:    9  (SWC-109)
    Unchecked External Calls      contracts:    31  functions:   57  (SWC-104)
  --------------------------------------------------
  Contracts with ALL 7 vulnerabilities:  1
============================================================
```

So you get: total valid contracts, total functions, per-vulnerability contract/function counts, and how many contracts have all configured vulnerability types.

## Empirical Limitation Checks

Use `evaluate_limitations.py` to generate code-based evidence for common tooling and labeling limitations.

### 1. Solidity 0.8+ overflow false-positive candidates

This checks how many contracts with `pragma >=0.8.0` are still labeled for integer overflow, and how many of those do not contain an `unchecked` block.

```cmd
py evaluate_limitations.py overflow-08 --from-json standardized_dataset\standardized_dataset.json
```

### 2. Reentrancy over-labeling with guards

This checks how many contracts/functions labeled as reentrant also contain `nonReentrant` or `ReentrancyGuard`.

```cmd
py evaluate_limitations.py reentrancy-guards --from-json standardized_dataset\standardized_dataset.json
```

### 3. Clean-contract count

This reports how many valid contracts have zero vulnerability labels.

```cmd
py evaluate_limitations.py clean-contracts --from-json standardized_dataset\standardized_dataset.json
```

### 4. Combined summary

Run all JSON-based limitation checks together:

```cmd
py evaluate_limitations.py summary --from-json standardized_dataset\standardized_dataset.json -o limitation_summary.txt
```

### 5. Scalability benchmark

Benchmark the current pipeline on increasing subset sizes:

```cmd
py evaluate_limitations.py benchmark contract_dataset_ethereum --sizes 100 500 1000 --fallback-only
```

You can also benchmark SmartBugs Wild:

```cmd
py evaluate_limitations.py benchmark smartbugs_wild\contracts --sizes 100 500 1000 --fallback-only --no-validate --no-dedup
```

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
