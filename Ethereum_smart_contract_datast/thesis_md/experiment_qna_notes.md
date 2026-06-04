# Experiment Q&A Notes

This file collects the main questions raised during the experiment phase and the concise answers used to justify design choices in the thesis.

## Why was CodeBERT used as the transformer model?

`CodeBERT` was chosen because it is a strong and widely accepted transformer model for source-code understanding. It is pretrained on code-related data, making it more suitable for Solidity vulnerability classification than general-language transformers such as `BERT` or `RoBERTa`. It also provides a clean `Transformer-only` semantic baseline for comparison against `GNN-only` and `Hybrid` models without introducing extra structural mechanisms that would blur the ablation design.

## Could another transformer have been used instead of CodeBERT?

Yes. Reasonable alternatives include:

- `GraphCodeBERT`
- `UniXcoder`
- `CodeT5`

Among these, `GraphCodeBERT` is the closest direct alternative. However, `CodeBERT` remains easier to justify as a clean semantic-only baseline because the hybrid model already adds structural information through AST/CFG graphs.

## Why is the hybrid model better than the unimodal models on the primary dataset?

The hybrid model is better mainly because it combines complementary semantic and structural information using a fusion technique that preserves the strengths of both inputs. `CodeBERT-only` captures token-level and contextual semantics, while `GNN-only` captures AST/CFG structural relationships. The hybrid model uses a gated, text-anchored fusion strategy so that structural information can improve the semantic representation when useful, rather than replacing it. Therefore, the gain comes from both the richer input and the fusion technique, with the fusion design being the more important factor.

## Why are Uninitialized Storage Pointer and Unchecked External Calls weak?

These labels are weak mainly because they are extremely sparse and highly imbalanced. They have very few positive examples compared with dominant classes such as reentrancy or transaction-ordering dependence, which makes stable learning and evaluation difficult. In addition, their patterns are more subtle and often require fine-grained reasoning about specific coding behavior rather than broad semantic clues. As a result, the models learn them less reliably and the metrics become much more sensitive to small changes in predictions and thresholds.

## What datasets were used in the experiment?

Two datasets were used:

- `ESC (Ethereum Smart Contracts dataset)` as the **primary dataset**
- `SmartBugs Wild` as the **secondary dataset**

The primary dataset was used for the full main experiment workflow, including `CodeBERT`, `GNN-only`, and `Hybrid` runs. The secondary dataset was used for supplementary validation and ablation analysis.

## Does a missing or zero result on some vulnerabilities mean the secondary dataset failed?

No. A zero or missing result usually means that the label is too rare in the evaluation split, or that the model did not predict enough positives for a stable F1 score. This is a support and imbalance issue rather than a dataset failure.

## Can the ESC result be treated as the result for both datasets?

No. The `ESC` result must be reported only as the result on `ESC`. It cannot be claimed as the result of both `ESC` and `SmartBugs Wild`. The secondary dataset must be described separately, even if the hybrid was only reduced-scale there.

## Why was SmartBugs Wild not run at full hybrid scale?

SmartBugs Wild is much larger than ESC in terms of total functions and contracts, and the AST/CFG graph extraction step becomes extremely expensive on this dataset. In the available Colab environment, full-scale structural and hybrid runs were computationally impractical. Therefore, the hybrid and GNN ablations on SmartBugs Wild were carried out at reduced scale, while the full main hybrid result was completed on the primary ESC dataset.

## What is the final recommended thesis result structure?

Use the following structure:

- `ESC` as the full main experimental dataset:
  - `CodeBERT-only`
  - `GNN-only`
  - `Hybrid`
- `SmartBugs Wild` as the secondary supplementary dataset:
  - `CodeBERT-only` at larger scale
  - `GNN-only` at reduced scale
  - `Hybrid` at reduced scale

This keeps the thesis methodologically honest while still showing that the proposed approach was evaluated on both datasets.

## What is the best final model result?

The strongest overall main result is:

- `ESC Hybrid recall_push_100k`

Key headline metrics:

- `Micro F1 = 0.9665`
- `Weighted F1 = 0.9672`
- `Subset Accuracy = 0.9839`

This should be treated as the final primary-dataset hybrid result.
