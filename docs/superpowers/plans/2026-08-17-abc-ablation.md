# Controlled A/B/C Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and execute a controlled A/B/C feature ablation before S4 evaluation and final five-subject training.

**Architecture:** A dedicated experiment module owns variant preprocessing, signal construction, TSFEL configuration, cache signatures, artifact persistence, and inference. The existing model/LOSO utilities remain shared. An English notebook runs A, B, and C in a fixed order and locks the winner before loading S4 labels.

**Tech Stack:** Python 3.13, pandas, NumPy, TSFEL, scikit-learn, joblib, nbformat, matplotlib, seaborn, pytest.

## Tasks

### Task 1: Variant Contracts
- Add tests for A/B/C signal sets, preprocessing equality for A/B, C robustness, and domain selection.
- Implement `abc_experiment_pipeline.py` definitions and signal builders.
- Run focused and full tests.

### Task 2: Variant Extraction, Cache, and Selection
- Add versioned per-variant cache APIs.
- Add labeled/unlabeled extraction and artifact inference APIs.
- Add deterministic winner selection using abnormal F1, worst-subject accuracy, and pooled macro-F1.
- Test cache isolation and selection.

### Task 3: English A/B/C Notebook
- Generate independent A/B/C LOSO sections with fixed seed 42.
- Print per-subject metrics, confusion matrices, hard-subject analysis, and A-to-B/B-to-C deltas.
- Lock the winner before the S4 section.
- Add S4 evaluation, final five-subject LOSO, and final artifact export.

### Task 4: Execution and Verification
- Extract fresh A/B/C caches for S1/S2/S3/S5.
- Execute the notebook end-to-end.
- Verify the winner, S4 error, five-subject results, artifacts, English-only source, and all tests.
