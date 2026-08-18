# V7-Compatible English Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an English-only, corrected v7-compatible TSFEL + HistGradientBoosting LOSO notebook and regenerated model/test outputs.

**Architecture:** Extend the reusable pipeline with leakage-safe v7 frame-level signals and curated statistical/temporal/spectral TSFEL features. Version feature caches, retain independent LOSO fold APIs, and generate an explicit notebook whose sections mirror v7 before final training and S4 prediction.

**Tech Stack:** Python 3.13, pandas, NumPy, SciPy, TSFEL 0.2.0, scikit-learn 1.7.2, joblib, pytest, nbformat, matplotlib, seaborn.

## Global Constraints

- Training subjects are exactly 1, 2, 3, and 5; shared test participant is 4.
- Predict exactly the eight official English activity classes.
- Use 30 FPS, 150-frame windows, training stride 75, test stride 150, and majority threshold 0.70.
- Do not require external Custom-domain files and do not reuse stale feature caches.
- All notebook prose, comments, output headings, and plot labels are English-only.

---

### Task 1: V7-Compatible Feature Signals and TSFEL Domains

**Files:**
- Modify: `tests/test_tsfel_histgb_pipeline.py`
- Modify: `tsfel_histgb_pipeline.py`

**Interfaces:**
- Produces: `FEATURE_SCHEMA_VERSION`, expanded `add_pose_signals(df)`, and `make_tsfel_config()` with statistical, temporal, and spectral domains.

- [ ] Add failing tests that require v7 behavior-oriented signals and a non-empty spectral TSFEL domain.
- [ ] Run the selected tests and confirm they fail against the reduced v8 feature set.
- [ ] Implement finite frame-level feature families without subject-wide transforms.
- [ ] Add curated built-in spectral features and keep deterministic feature ordering.
- [ ] Run the feature and full unit suites.

### Task 2: Versioned Feature Cache

**Files:**
- Modify: `train_tsfel_histgb.py`
- Modify: `tests/test_tsfel_histgb_pipeline.py`

**Interfaces:**
- Cache entries include source signature, `FEATURE_SCHEMA_VERSION`, and a deterministic TSFEL configuration signature.

- [ ] Add a test proving a cache with the previous feature schema is rejected.
- [ ] Implement cache metadata generation and validation.
- [ ] Run the cache test and compilation checks.

### Task 3: English V7-Compatible Notebook

**Files:**
- Modify: `build_tsfel_notebook.py`
- Generate: `ISAS_CHALLENGE_full_pipeline_v8_TSFEL_FIXED.ipynb`

**Interfaces:**
- Consumes the module and cache APIs.
- Produces explicit English sections for loading, normalization, handcrafted features, TSFEL, four LOSO folds, final fit, artifact reload, and S4 prediction.

- [ ] Rewrite every markdown cell, code comment, print heading, and plot label in English.
- [ ] Show the v7-compatible engineered feature families and preview before extraction.
- [ ] Keep four separate `run_loso_fold` cells with independent outputs.
- [ ] Generate and validate the notebook schema and Python syntax.

### Task 4: Full Regeneration and Verification

**Files:**
- Regenerate: `artifacts/tsfel_histgb/*`
- Regenerate: `outputs/tsfel_histgb/*`
- Execute: `ISAS_CHALLENGE_full_pipeline_v8_TSFEL_FIXED.ipynb`

**Interfaces:**
- Produces fresh feature caches, LOSO metrics/confusion matrices, joblib artifact, and S4 predictions.

- [ ] Force feature re-extraction so no reduced-v8 cache survives.
- [ ] Execute the notebook with the Anaconda base kernel.
- [ ] Verify every code cell executed without an error and all four fold cells contain metrics/report/heatmap outputs.
- [ ] Verify artifact schema, feature order, output row counts, eight-class labels, and exact submission columns.
- [ ] Run the complete pytest suite and Python compilation as the final gate.
