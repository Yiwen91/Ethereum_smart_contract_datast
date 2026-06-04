# Thesis Markdown Chapters

| Chapter | File | Content |
|---------|------|---------|
| 1 | `chapter_01_introduction.md` | Background, motivation, RQs, scope |
| 2 | `chapter_02_literature_review.md` | Literature and research gaps |
| 3 | `chapter_03_research_methodology.md` | Slither labeling, hybrid GCN, experiments |
| 4 | `chapter_04_results_and_discussion.md` | **Experimental results** (ESC + SmartBugs) |
| 5 | `chapter_05_conclusion.md` | Contributions, limitations, future work |

Supporting notes: `experiment_qna_notes.md`, `smartbugs_tuning_summary.md`, `case_study_report.md`.

**Colab (Slither splits, all ablations):** `colab_end_to_end_slither_experiments.md`

**Labeling:** `standardize_dataset.py --labeler slither`  
**Training:** `train_experiment.py --model codebert|gnn|hybrid|slither`  
**Cross-contract:** `--hybrid-enable-cross-contract`  
**Explainability:** `run_shap_explain.py`
