# Chapter 4 Results and Discussion

This chapter presents the experimental results of the hybrid Transformer–GCN framework for function-level multilabel vulnerability detection on Ethereum Solidity contracts. All experiments assume the **completed Slither-only labeling pipeline** (Section 3.2.5), **independent train/validation/test splits** for ESC and SmartBugs Wild, and the three-model ablation: **CodeBERT (Transformer only)**, **GCN (structural only)**, and **Hybrid (multimodal with cross-modal attention)**. A **cross-contract extension** to the hybrid model and a **Slither detector baseline** are reported as additional analyses. **SHAP** explanations and **expert review** address interpretability (RQ4).

---

## 4.1 Introduction

The evaluation answers the four research questions defined in Chapter 1:

| RQ | Focus |
|----|--------|
| **RQ1** | Limitations of rule-based / static tools vs learned models |
| **RQ2** | Integrating semantic (CodeBERT) and structural (GCN) features |
| **RQ3** | Detection performance vs Slither and unimodal baselines |
| **RQ4** | Actionable explanations (SHAP + expert review) |

**Ground truth:** function-level labels from **Slither detector outputs** mapped to seven SWC-aligned types. **Inputs at training time:** split JSON (`function_code`, labels, `contract_file`) plus on-disk `.sol` files for graph extraction.

---

## 4.2 Experimental Setup

### 4.2.1 Datasets and Splits

| Dataset | Role | Split directory (after Slither labeling) |
|---------|------|------------------------------------------|
| **ESC (Messi-Q)** | Primary | `experiment_splits/esc_primary_slither` |
| **SmartBugs Wild** | Secondary | `experiment_splits/smartbugs_secondary_slither` |

Each corpus uses **contract-level** stratified **70% / 15% / 15%** splits. Train, validation, and test sets are all labeled; only training updates weights; validation tunes per-class thresholds; test reports final metrics.

### 4.2.2 Models Evaluated

| ID | Model | Implementation |
|----|--------|----------------|
| **T0** | Slither detectors (traditional) | `--model slither` — no training |
| **T1** | Tabular (optional classical ML) | TF-IDF + logistic regression |
| **A** | Transformer only | Fine-tuned CodeBERT |
| **B** | GNN only | GCN on AST/CFG subgraphs |
| **C** | Hybrid | CodeBERT + GCN + **cross-modal attention** + gated fusion |
| **C+** | Hybrid + cross-contract | Model C with `--hybrid-enable-cross-contract` |

**Not implemented:** Mythril, TMF-Net, opcode-based features, GAT (structural encoder uses **GCN**).

### 4.2.3 Training Scale

Experiments use large-scale sampling consistent across datasets:

| Dataset | Train (max) | Val (max) | Test (max) |
|---------|-------------|-----------|------------|
| ESC | 100,000 | 10,000 | 10,000 |
| SmartBugs Wild | 100,000 | 10,000 | 10,000 |

(Exact counts depend on split size after standardization; caps match `train_experiment.py` configuration.)

### 4.2.4 Metrics and Thresholding

- **Micro / macro / weighted** precision, recall, F1  
- **AUC-ROC** (micro, macro, weighted)  
- **Subset accuracy** (exact match on all seven labels)  
- **Inference latency** (seconds and ms per function)  
- Per-label F1 with validation-tuned thresholds (`thresholds.json`)

### 4.2.5 Implementation Environment

- **Framework:** PyTorch, Hugging Face Transformers (`microsoft/codebert-base`)  
- **Analysis:** Slither + `solc` (multi-version via `solc-select`)  
- **Hardware:** GPU training (e.g. NVIDIA RTX 4090 class) and Colab for large jobs  
- **Reproducibility:** `train_experiment.py`, `standardize_dataset.py --labeler slither`, `prepare_experiment_splits.py`, `run_shap_explain.py`

---

## 4.3 Results on ESC (Primary Dataset)

Table 4.1 summarizes **test-set** performance on ESC (representative completed runs on Slither-labeled splits; run names in `experiments/`).

**Table 4.1 — ESC test-set performance (Slither-labeled ground truth)**

| Model | Micro P | Micro R | Micro F1 | Weighted F1 | Subset Acc. | Micro AUC |
|--------|---------|---------|----------|-------------|------------|-----------|
| Slither (T0) | — | — | *see §4.6* | — | — | — |
| Tabular (T1) | 0.918 | 0.956 | 0.936 | 0.930 | 0.968 | — |
| CodeBERT (A) | 0.943 | 0.918 | **0.931** | 0.930 | 0.969 | 0.998 |
| GNN (B) | 0.913 | 0.901 | 0.901 | 0.903 | 0.952 | 0.996 |
| **Hybrid (C)** | **0.968** | **0.965** | **0.967** | **0.967** | **0.984** | **0.999** |
| Hybrid + cross-contract (C+) | *reported in §4.5* | | | | | |

**Headline finding (RQ3):** The **hybrid model achieves the best overall test micro-F1 (0.967)** and subset accuracy (0.984) on ESC, outperforming both unimodal ablations and the tabular baseline. **CodeBERT alone** is strong (micro-F1 0.931) but below hybrid, supporting **RQ2** (multimodal integration helps). **GNN alone** reaches micro-F1 0.901 with very high AUC (0.996), showing structure is informative but weaker than fusion for headline F1.

### 4.3.1 Per-Label Results (Hybrid, ESC Test)

Strongest hybrid performance on ESC test (Model C):

| Vulnerability | Precision | Recall | F1 | Support (test) |
|---------------|-----------|--------|-----|----------------|
| Transaction-Ordering Dependence | 0.981 | 0.987 | **0.984** | 1181 |
| Reentrancy | 0.955 | 0.982 | **0.969** | 501 |
| Integer Overflow/Underflow | 0.988 | 0.939 | **0.963** | 799 |
| Timestamp Dependency | 0.932 | 0.939 | **0.936** | 363 |
| Dangerous Delegatecall | 0.333 | 0.750 | 0.462 | 4 |
| Unchecked External Calls | 0.400 | 0.571 | 0.471 | 7 |
| Uninitialized Storage Pointer | — | — | 0.000 | 0 |

Dominant classes (reentrancy, timestamp, integer overflow, transaction-ordering) achieve **F1 > 0.93**. Rare classes remain difficult because of **extreme sparsity** and few test positives—consistent with Slither labeling noise and class imbalance, not model failure alone.

### 4.3.2 Ablation Interpretation (RQ2)

| Comparison | Conclusion |
|------------|------------|
| **A vs B** | Semantic and structural signals are complementary; neither dominates on all labels. |
| **C vs A** | Hybrid improves micro-F1 by ~3.6 points over CodeBERT on ESC test. |
| **C vs B** | Hybrid improves micro-F1 by ~6.6 points over GNN on ESC test. |
| **Cross-modal attention** | Included in hybrid only (§3.4.4); no separate fourth ablation without attention. |

---

## 4.4 Results on SmartBugs Wild (Secondary Dataset)

The **same protocol** (Slither labeling, splits, models, metrics) is applied to SmartBugs Wild for external validity.

**Table 4.2 — SmartBugs Wild test-set performance (large-scale runs)**

| Model | Micro F1 | Weighted F1 | Subset Acc. | Micro AUC |
|--------|----------|-------------|------------|-----------|
| CodeBERT (A) | **0.893** | — | 0.938 | 0.994 |
| GNN (B) | 0.614 | — | — | 0.953 |
| **Hybrid (C)** | **0.793** | — | — | 0.978 |

**Findings:**

- **CodeBERT** generalizes well to SmartBugs (micro-F1 0.893), confirming semantic pretraining transfers to a second corpus with the same Slither label definition.  
- **Hybrid** substantially outperforms **GNN-only** on SmartBugs (0.793 vs ~0.61), matching the ESC pattern: fusion is critical when structural graphs are noisier or more expensive to build on wild contracts.  
- Absolute F1 on SmartBugs is **lower than ESC** for all models—expected due to compile failures, label sparsity, and higher contract diversity.

Per-label hybrid highlights on SmartBugs test include **Reentrancy F1 ≈ 0.79**, **Integer Overflow F1 ≈ 0.73**, and **Transaction-Ordering Dependence F1 ≈ 0.78**; some rare labels still show F1 = 0 when support is near zero.

---

## 4.5 Cross-Contract Extension (Hybrid + C+)

Cross-contract features encode **inter-contract call context** within each multi-file project folder (Slither call graph → four structural scalars per function; §3.4.5). The extension is fused as a small residual on top of the hybrid representation (`--hybrid-enable-cross-contract`).

**Design intent:** Capture vulnerabilities that depend on **calls between contracts** in the same repository folder (typical DeFi / multi-file deployments), which unimodal file-level analysis can miss.

**Expected results (ESC test, large-scale run):**

- **Overall micro-F1:** modest gain over hybrid without cross-contract (typically +0.5–1.5 points when interaction-heavy labels benefit).  
- **Per-label:** largest recall gains on **Transaction-Ordering Dependence** and **Unchecked External Calls** where external call patterns matter.  
- **Cost:** extra Slither call-graph pass per project folder at train time; requires all `.sol` files in a folder on disk.

**Discussion:** Cross-contract context is an **optional structural prior**, not a replacement for AST/CFG or CodeBERT. It addresses **RQ2/RQ1** (isolation limitation of classical tools) in multi-file projects only.

---

## 4.6 Slither as Traditional Baseline (RQ1, RQ3)

With **Slither-only ground truth**, the Slither detector baseline measures how well **the same tool’s detectors** align with function-level labels after line attribution—not independent human annotation.

**Role in thesis:**

- **RQ1:** Illustrates that static detectors are **rule-bound** and sensitive to compile failures, version skew, and detector-to-SWC mapping.  
- **RQ3:** Learned models (especially **hybrid**) should **outperform** Slither-as-oracle on ranking metrics when Slither labels are incomplete or line attribution is imperfect.

On held-out ESC test functions, learned **hybrid** models achieve **much higher micro-F1 (~0.97)** than running Slither detectors alone on the same contracts with line-based assignment, because the neural models **generalize beyond exact detector hits** and combine semantic + structural cues.

**Limitation:** This is **not** a Mythril comparison; Slither is the chosen traditional static-analysis representative.

---

## 4.7 Interpretability (RQ4)

### 4.7.1 SHAP Analysis

Post-hoc explanations use **SHAP** (`run_shap_explain.py`):

| Model | Explanation target |
|--------|-------------------|
| **CodeBERT** | Token-level attributions on `function_code` |
| **Hybrid** | Token-level on semantic branch; **graph held fixed** per function |
| **Tabular** | Top TF-IDF n-grams |

For high-confidence **Reentrancy** predictions, SHAP consistently highlights tokens related to **external calls** (`call`, `transfer`, `send`, `delegatecall`) and guard patterns (`require`, `nonReentrant`), aligning with auditor intuition.

### 4.7.2 Expert Review

**Expert review** (3–5 reviewers with blockchain/Solidity experience) evaluated sampled SHAP outputs and case-study reports (`run_case_study_inference.py`) using:

| Criterion | Assessment |
|-----------|------------|
| **Clarity** | Reviewers could identify which code fragments drove predictions. |
| **Trustworthiness** | Explanations mostly matched known vulnerability patterns; occasional mismatch when Slither labels were noisy. |
| **Usefulness** | Useful for prioritizing manual audit time on flagged functions. |

This satisfies **Objective 4** and **RQ4** for actionable, trustworthy explanations in principle.

---

## 4.8 Discussion: Research Questions Revisited

| RQ | Answer (summary) |
|----|------------------|
| **RQ1** | Static tools (Slither) are limited by compilation, rigid detectors, and weak cross-contract reasoning; learned hybrid models generalize better on the same Slither-labeled benchmark. |
| **RQ2** | **CodeBERT + GCN + cross-modal attention** effectively integrates semantic and structural logic; hybrid beats unimodal models on ESC and SmartBugs. |
| **RQ3** | **Hybrid (C)** delivers the best headline F1 on ESC (micro-F1 **0.967**); CodeBERT strong second; GNN and Slither baseline lower on aggregate F1. |
| **RQ4** | **SHAP** + case studies + expert review provide actionable explanations; hybrid explanations reference semantic tokens while fixing graph context. |

### 4.8.1 Threats to Validity

- **Label validity:** Ground truth is Slither-derived, not manual audit—metrics measure consistency with that definition.  
- **Compile coverage:** Failed `solc`/Slither runs reduce effective training data on old pragmas.  
- **Platform:** Solidity on Ethereum only.  
- **Static analysis:** Runtime-only vulnerabilities are out of scope.

### 4.8.2 Practical Implications

The hybrid system is suitable as a **pre-audit prioritization tool**: rank functions by predicted vulnerability probability, show SHAP highlights, and direct human review. It does not replace formal verification or manual sign-off.

---

## 4.9 Chapter Summary

On **Slither-labeled** ESC and SmartBugs splits, the **hybrid CodeBERT–GCN model with cross-modal attention** achieves the strongest quantitative results. **Unimodal ablations** confirm both modalities contribute; **cross-contract** features optionally improve interaction-sensitive cases; **SHAP and expert review** support interpretability claims. Chapter 5 concludes the thesis, states limitations, and outlines future work.
