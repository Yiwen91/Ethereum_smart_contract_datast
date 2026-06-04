# Chapter 3 Research Methodology

To provide a holistic overview of the research workflow, Figure 3.1 illustrates the end-to-end methodology, encompassing data collection, preparation, model development, experimental design, and evaluation. This diagram synthesizes the core phases of the study, aligning with the research objectives to develop and validate the hybrid Transformer-GNN framework.

## 3.1 Introduction

This chapter details the methodology used to achieve the research objectives and address the research questions. The approach adopts a sequential explanatory mixed-methods design, combining quantitative model development and evaluation with qualitative interpretability assessment. The methodology is structured to ensure rigor, replicability, and practical relevance, focusing on the development of a hybrid Transformer-GNN framework for Solidity smart contract vulnerability detection.

The five core steps are structured as follows:

1. **Data Collection**
   Curates a comprehensive dataset to ensure diversity, representativeness, and real-world relevance.
2. **Data Preparation and Preprocessing**
   Converts raw Solidity code into model-compatible inputs for semantic and structural analysis.
3. **Model Development**
   Builds the hybrid Transformer-GNN framework and integrates interpretability modules.
4. **Experimental Design**
   Defines baselines and ablation studies for validating the framework.
5. **Evaluation**
   Assesses quantitative performance and qualitative interpretability.

This workflow is intended to address the research gaps identified in Chapter 2 while maintaining methodological rigor and alignment with the study’s objectives.

## 3.2 Dataset

To ensure generalizability and robustness, the study uses a combination of benchmark datasets and real-world contracts.

### 3.2.1 Primary Dataset: Ethereum Smart Contracts (ESC) Dataset

The primary dataset consists of 30,124 real-world Ethereum smart contracts sourced from the Ethereum mainnet and curated by Liu et al. (2023). It is publicly available via the Messi-Q research repository.

Key features include:

- **Size and Diversity:** 30,124 valid Solidity contracts and 646,346 functions, spanning DeFi, NFT, governance, and utility use cases.
- **Labeling:** Seven vulnerability types assigned at function level using **Slither detector outputs** mapped to SWC-aligned categories (see Section 3.2.5):
  - Reentrancy (SWC-107)
  - Integer Overflow/Underflow (SWC-101)
  - Unchecked External Calls (SWC-104)
  - Dangerous Delegatecall (SWC-112)
  - Transaction-Ordering Dependence (SWC-114)
  - Timestamp Dependency (SWC-116)
  - Uninitialized Storage Pointer (SWC-109)
- **Ethereum Relevance:** Covers Solidity versions 0.4.x to 0.8.x and focuses on contracts with real mainnet activity.

The ESC dataset is presented as a strong benchmark for real-world Ethereum vulnerability detection and is cited by several prior studies for evaluating both traditional and deep learning approaches.

Reference:

- `https://github.com/Messi-Q/Smart-Contract-Dataset`

### 3.2.2 Secondary Dataset: SmartBugs Wild

SmartBugs Wild is a large-scale and widely recognized benchmark dataset for smart contract vulnerability detection, initially curated by Durieux et al. (2020).

Key features include:

- **Scale:** 47,398 Solidity files containing 203,716 contracts.
- **Labeling:** The same **Slither-only labeling protocol** as ESC is applied so both datasets share one ground-truth definition (seven SWC-aligned types at function level).
- **Vulnerability Coverage:** Supports evaluation across the same seven vulnerability classes used on ESC, enabling comparable multilabel metrics between primary and secondary benchmarks.
- **Real-World Relevance:** Includes diverse contract categories such as DeFi, NFTs, governance, and general-purpose applications.

Several studies cited in the chapter use SmartBugs Wild as a benchmark for evaluating multimodal and unimodal vulnerability detection models.

Reference:

- `https://github.com/smartbugs/smartbugs-wild`

### 3.2.3 Rationale for Dataset Selection

The dual-dataset approach is justified by two research goals: real-world generalizability and direct comparability with state-of-the-art models.

#### ESC Dataset

- Aligns with the target Ethereum ecosystem, especially legacy pre-0.8.x contracts.
- Covers Ethereum-specific and understudied vulnerability types.
- Supports training on contracts that better reflect real production environments.

#### SmartBugs Wild

- Enables secondary benchmarking and generalization testing beyond the Messi-Q-derived ESC corpus.
- Uses the **same experimental protocol** as ESC (labeling, splits, models, and metrics) for fair comparison.
- Serves as a widely cited benchmark in the smart contract security literature.

### 3.2.4 Dataset Splitting

The ESC and SmartBugs corpora are **not merged**. Each dataset is split **independently** at **contract level** using the same ratio:

- **Training set (70%)**
- **Validation set (15%)**
- **Test set (15%)**

Splitting is **stratified by vulnerability label sets** so rare types appear across partitions where possible. All functions from the same contract remain in the same split to avoid leakage.

Implementation outputs (after Slither labeling):

- **Primary:** `experiment_splits/esc_primary_slither` (or equivalent) from standardized ESC JSON
- **Secondary:** `experiment_splits/smartbugs_secondary_slither` from standardized SmartBugs JSON

Train, validation, and test partitions **all contain labels**. Only the training set updates model parameters; validation is used for threshold tuning and checkpoint selection; the test set is reserved for final reported metrics.

### 3.2.5 Slither-Only Labeling Protocol

Ground truth is produced by the project standardization pipeline (`standardize_dataset.py` with `--labeler slither`):

1. **Function extraction** — Slither parses each `.sol` file (regex fallback only when parsing fails).
2. **Contract-level analysis** — Slither runs its built-in detectors on the compilable contract.
3. **Detector-to-SWC mapping** — Each finding is mapped to one of the seven thesis vulnerability types using a fixed check-name table (`slither_labeling.py`).
4. **Function attribution** — A vulnerability is assigned to a function when a finding’s source line falls within that function’s line range.
5. **Export** — Results are stored in `standardized_dataset.json` (function code, labels, `swc_ids`, and metadata `labeling_source: slither`).

Contracts that fail to compile are skipped or receive empty Slither labels for that file; this limitation is reported in evaluation. The same protocol is applied to **ESC** and **SmartBugs Wild** before split generation (`prepare_experiment_splits.py`).

## 3.3 Technical Tools

This section describes the tools and resources used for data preprocessing, model development, and evaluation.

### 3.3.1 Data Preprocessing Tools

- **Slither:** Parses Solidity source code, runs detectors for **labeling**, and supports AST/CFG extraction for graphs.
- **Solidity Compiler (`solc`):** Required for Slither compilation (multiple `solc` versions via `solc-select` for legacy `pragma` lines).
- **Python Libraries:** NumPy and custom graph utilities are used for tensor and graph processing.

### 3.3.2 Model Development Tools

- **PyTorch:** Used to implement the Transformer, GCN, and hybrid fusion components.
- **Hugging Face Transformers:** Provides pre-trained CodeBERT models.
- **Scikit-learn:** Used for the optional tabular baseline and metric computation.

### 3.3.3 Interpretability Tools

- **SHAP:** Generates localized explanations for predictions.
- **Matplotlib / Seaborn:** Used to visualize attention weights and SHAP outputs.

### 3.3.4 Hardware and Software

- **Hardware:** GPU-enabled workstation with NVIDIA RTX 4090.
- **Software:** Ubuntu 22.04 LTS, Python 3.9+, and Jupyter Notebooks.

## 3.4 Proposed Framework

The proposed framework integrates Transformer-based semantic encoding with GNN-based structural reasoning to detect smart contract vulnerabilities.

### 3.4.1 Stage 1: Data Input

The input consists of Solidity smart contract source code collected from the datasets described earlier.

### 3.4.2 Stage 2: Preprocessing

Two parallel preprocessing pipelines feed the hybrid model:

- **Semantic preprocessing:** Function source is tokenized with the CodeBERT tokenizer (`function_code` from the standardized JSON).
- **Structural preprocessing:** Slither-derived AST/CFG information is converted into a **function-local subgraph** (bounded node budget) for GNN input.

Opcode-level bytecode features are **not** used in this study; structural signal comes from AST/CFG graphs only.

### 3.4.3 Stage 3: Feature Extraction

- **Transformer module:** Fine-tuned CodeBERT (`microsoft/codebert-base`) produces a semantic embedding from each function’s source text.
- **GNN module:** A **graph convolutional network (GCN)** over the AST/CFG subgraph produces a structural embedding (mean/max pooling over nodes).

### 3.4.4 Stage 4: Feature Fusion

The hybrid model fuses modalities with:

1. Linear projection of semantic and structural vectors to a shared fusion dimension.
2. **Cross-modal multi-head attention** (text query over semantic and structural modality tokens).
3. **Gated residual fusion** using concatenated text, graph, difference, and element-wise interaction features.

This cross-modal attention is **part of the hybrid model**; it is not evaluated as a separate fourth ablation.

### 3.4.5 Stage 5: Vulnerability Classification, Cross-Contract, and Interpretability

- **Classification:** A multilabel head outputs seven vulnerability probabilities per function; per-class thresholds are tuned on the validation split.
- **Cross-contract extension (hybrid):** An optional branch encodes **inter-contract call context** within each multi-file project folder (Slither call graph, four structural features per function) and adds a small residual to the hybrid fusion. Reported results include hybrid **with** and **without** this flag where applicable.
- **Interpretability:** **SHAP** (token-level on the semantic branch; graph held fixed per function for hybrid) and **case-study reports** on individual contracts support qualitative analysis; expert review assesses clarity, trustworthiness, and usefulness.

### 3.4.6 Stage 6: Final Output

The framework delivers actionable results for auditors and developers, including:

- vulnerability type labels
- confidence scores
- vulnerability locations
- interpretability summaries
- cross-contract context where applicable

### 3.4.7 Framework Mechanism for Ethereum Smart Contract

The framework is designed specifically for Ethereum and Solidity by combining:

- **Ethereum-aware input processing**
- **Solidity-oriented semantic feature encoding**
- **EVM-centered structural feature learning**
- **context-aware cross-modal fusion**
- **Ethereum-tailored vulnerability classification**
- **interpretability for Ethereum security audits**

This design aims to better detect Ethereum-native vulnerabilities such as reentrancy, delegatecall misuse, timestamp dependency, and uninitialized storage pointer issues.

## 3.5 Experimental Design

### 3.5.1 Baseline Comparisons

Performance is reported on **both datasets** (ESC primary, SmartBugs secondary) using the **same Slither-labeled splits** and metrics.

| Category | Implementation in this study |
|----------|------------------------------|
| **Traditional static analysis** | **Slither detectors** mapped to the seven labels (`train_experiment.py --model slither`) — no training, function-level assignment by line range |
| **Classical ML (optional)** | TF-IDF + logistic regression (`tabular`) |
| **Deep learning — unimodal** | CodeBERT only; GCN on AST/CFG only |
| **Deep learning — proposed** | Hybrid (CodeBERT + GCN + cross-modal attention); optional **hybrid + cross-contract** |

Mythril, TMF-Net, and other external architectures cited in the literature are discussed in Chapter 2 but **not re-implemented** here; **Slither** is the traditional-tool comparator.

### 3.5.2 Ablation Study

Three ablation models isolate modality contributions (cross-modal attention is **included in the hybrid** model only):

| Model | Role | Implementation |
|-------|------|----------------|
| **Model A — Transformer only** | Semantic baseline | `--model codebert` |
| **Model B — GNN only** | Structural baseline | `--model gnn` |
| **Model C — Hybrid** | Full multimodal model (CodeBERT + GCN + cross-modal attention + gated fusion) | `--model hybrid` |

The hybrid may additionally be run with `--hybrid-enable-cross-contract` to measure the cross-contract extension. There is **no separate ablation** that removes cross-modal attention while keeping fusion; unimodal vs hybrid comparison addresses multimodal benefit.

### 3.5.3 Evaluation Metrics

Both quantitative and qualitative metrics are used.

#### Quantitative Metrics

- Precision
- Recall
- F1-score
- AUC-ROC
- Accuracy
- Inference Latency

These metrics are defined in terms of true positives, true negatives, false positives, and false negatives.

#### Qualitative Metrics

Interpretability is evaluated through expert reviews with 3 to 5 blockchain developers using:

- **Clarity**
- **Trustworthiness**
- **Usefulness**

Feedback is collected via semi-structured interviews and structured surveys to assess whether the interpretability outputs are understandable, faithful, and practically useful in auditing workflows.

## 3.6 Summary

This chapter outlined the methodology: **Slither-only** function-level labeling on **ESC** and **SmartBugs Wild**, **separate** stratified 70/15/15 contract splits, a **CodeBERT + GCN hybrid** with cross-modal attention, **Slither** as the traditional baseline, a **three-model ablation** (Transformer, GNN, Hybrid), optional **cross-contract** hybrid runs, and **SHAP** plus expert review for interpretability. Quantitative evaluation uses the same metrics and threshold protocol on both datasets at large scale where resources allow.
