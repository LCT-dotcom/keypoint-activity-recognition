# Phase D Accuracy Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a leakage-safe Phase D champion-challenger pipeline that improves frame-level subject-independent activity accuracy without tuning on S4.

**Architecture:** New focused modules own frame aggregation/decoding, classical feature selection/model tuning, compact multi-scale extraction, and the optional shallow TCN. Existing A/B/C extraction and class contracts remain shared. A runner and English notebook orchestrate nested LOSO, lock one winner, perform secondary S4 analysis, then produce five-subject results and final artifacts.

**Tech Stack:** Python 3.13, pandas, NumPy, scikit-learn, TSFEL caches, joblib, PyTorch when available, nbformat, pytest.

## Global Constraints

- Development and model selection use S1, S2, S3, and S5 only.
- S4 labels never influence features, hyperparameters, fusion, decoding, architecture, or winner selection.
- Primary selection uses frame-level outer LOSO; legacy 150-frame window metrics remain diagnostic.
- Random state is 42.
- A candidate must preserve abnormal F1 within 0.5 percentage points, preserve worst-subject accuracy, and avoid any subject loss above 5 points versus D0-C.
- Eligible candidates rank by pooled frame accuracy, worst-subject accuracy, macro F1, then abnormal F1.
- If no candidate is eligible, keep C.
- Full searches are bounded, signed, cached, and resumable.
- This directory is not a Git repository, so verification checkpoints replace commit steps.

---

### Task 1: Common Frame Probability Evaluator

**Files:**
- Create: `phase_d_evaluation.py`
- Create: `tests/test_phase_d_evaluation.py`

**Interfaces:**
- Consumes: `CLASSES`, `ABNORMAL_CLASSES`, window metadata with `start_pos` and `end_pos`, and per-window class probabilities.
- Produces: `FrameProbabilityResult`, `aggregate_window_probabilities(...)`, `soft_group_fusion(...)`, `estimate_transition_model(...)`, `viterbi_decode(...)`, `evaluate_frame_result(...)`.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_overlapping_probabilities_are_averaged_and_edges_are_covered():
    meta = pd.DataFrame({"start_pos": [0, 2], "end_pos": [4, 6]})
    probabilities = np.array([[0.8, 0.2], [0.2, 0.8]])
    result = aggregate_window_probabilities(meta, probabilities, n_frames=6)
    np.testing.assert_allclose(result.probabilities[2:4], [[0.5, 0.5], [0.5, 0.5]])
    assert result.coverage.tolist() == [1, 1, 2, 2, 1, 1]
```

- [ ] **Step 2: Run RED test**

Run: `python -m pytest tests/test_phase_d_evaluation.py -q -p no:cacheprovider`

Expected: collection failure because `phase_d_evaluation` does not exist.

- [ ] **Step 3: Implement immutable result types and frame aggregation**

Implement strict shape/class-order validation, probability averaging, nearest-covered fallback for genuine gaps, and deterministic argmax labels.

- [ ] **Step 4: Run GREEN test and full suite**

Run focused test, then `python -m pytest tests -q -p no:cacheprovider`.

- [ ] **Step 5: Add RED tests for soft fusion and Viterbi**

```python
def test_soft_gate_reweights_but_never_removes_a_class():
    fused = soft_group_fusion(flat, gate, alpha=1.0, classes=classes, abnormal_classes={"B"})
    assert np.all(fused > 0)
    np.testing.assert_allclose(fused.sum(axis=1), 1.0)

def test_zero_transition_strength_matches_frame_argmax():
    decoded = viterbi_decode(emissions, transition, initial, strength=0.0)
    assert decoded.tolist() == emissions.argmax(axis=1).tolist()
```

- [ ] **Step 6: Implement fusion, transition estimation, Viterbi, and frame metrics**

Use epsilon-clipped log probabilities, Laplace-smoothed transitions, explicit class ordering, and `evaluate_predictions` for shared metrics.

- [ ] **Step 7: Verify Task 1**

Require focused and full tests to pass with no new warnings.

### Task 2: Training-Only Selector and Tuned HistGradientBoosting

**Files:**
- Create: `phase_d_classical.py`
- Create: `tests/test_phase_d_classical.py`

**Interfaces:**
- Consumes: `WindowFeatures`, subject ids, fixed parameter candidates, and seed.
- Produces: `TrainingOnlyFeatureSelector`, `ClassicalConfig`, `FittedClassicalModel`, `fit_classical_model(...)`, `nested_select_classical_config(...)`.

- [ ] **Step 1: Write RED leakage and schema tests**

```python
def test_selector_fit_uses_training_rows_only():
    selector = TrainingOnlyFeatureSelector(feature_budget=2, random_state=42)
    selector.fit(train_x, train_y)
    selected_before = selector.selected_columns
    altered_test_x = test_x * 1_000_000
    selector.transform(altered_test_x)
    assert selector.selected_columns == selected_before

def test_transform_rejects_missing_selected_columns():
    with pytest.raises(ValueError, match="missing selected features"):
        selector.transform(frame.drop(columns=[selector.selected_columns[0]]))
```

- [ ] **Step 2: Run RED test**

Expected: import failure because `phase_d_classical` does not exist.

- [ ] **Step 3: Implement selector**

Fit median imputation statistics, constant removal, deterministic Spearman 0.98 pruning, mutual-information ranking with seed 42, and stable feature order. Store all fitted state explicitly.

- [ ] **Step 4: Run selector tests GREEN**

Run focused tests and inspect selected column stability across two fits.

- [ ] **Step 5: Add RED weighted-fit and nested-boundary tests**

Test `none` versus square-root inverse-frequency weights and assert the outer held-out subject id never appears in an inner training/validation record.

- [ ] **Step 6: Implement configurable HistGradientBoosting and nested selection**

Use `ParameterSampler(n_iter=24, random_state=42)` over the exact spec grid. Score inner folds by eligibility-aware accuracy: reject abnormal-F1 regressions above tolerance, then rank mean accuracy, worst accuracy, macro F1, abnormal F1.

- [ ] **Step 7: Verify Task 2**

Run focused tests, all tests, and a small synthetic nested-LOSO smoke test.

### Task 3: Compact Multi-Scale Geometry

**Files:**
- Create: `phase_d_multiscale.py`
- Create: `tests/test_phase_d_multiscale.py`

**Interfaces:**
- Consumes: raw keypoint DataFrames and `normalize_for_experiment(..., "C")`.
- Produces: `build_compact_signals(...)`, `extract_compact_windows(...)`, `MultiScaleWindows`, `fuse_scale_probabilities(...)`, versioned cache helpers.

- [ ] **Step 1: Write RED signal-invariance tests**

```python
def test_compact_signals_include_bones_angles_velocity_and_acceleration():
    signals = build_compact_signals(normalized_pose)
    assert {"left_elbow_angle", "left_wrist_speed", "left_wrist_acceleration"} <= set(signals)
    assert np.isfinite(signals.to_numpy()).all()
```

- [ ] **Step 2: Run RED test**

Expected: import failure because `phase_d_multiscale` does not exist.

- [ ] **Step 3: Implement compact anatomical signals and summaries**

Use shared C normalization, symmetric bones/angles, velocities/accelerations, and the fixed summary list. Avoid full TSFEL and avoid participant-label statistics.

- [ ] **Step 4: Add RED window/cache tests**

Assert exact window sizes 60/150/300, majority thresholds, metadata, cache-signature isolation, and no cross-scale cache collision.

- [ ] **Step 5: Implement extraction and cache**

Use half-window training strides, inference strides 15/30, tail coverage, stable schemas, and source-file/config hashes.

- [ ] **Step 6: Add RED fusion-grid test and implement fusion**

Evaluate only the ten declared weight triples and reject mismatched frame lengths or class orders.

- [ ] **Step 7: Verify Task 3**

Run focused/full tests and extract a short real-data slice to verify finite features and label alignment.

### Task 4: Outer LOSO Champion with Hierarchical and Temporal Ablations

**Files:**
- Create: `phase_d_runner.py`
- Create: `tests/test_phase_d_runner.py`

**Interfaces:**
- Consumes: A/B/C C caches, compact multi-scale caches, Task 1 evaluator, Task 2 models, Task 3 fusion.
- Produces: `CandidateResult`, `OuterFoldResult`, `run_d0_fold(...)`, `run_d1_fold(...)`, `run_d2_fold(...)`, `run_d3_fold(...)`, `select_phase_d_winner(...)`.

- [ ] **Step 1: Write RED selection-rule tests**

```python
def test_ineligible_accuracy_gain_cannot_beat_baseline():
    winner = select_phase_d_winner([baseline, high_accuracy_low_abnormal])
    assert winner.name == baseline.name

def test_eligible_candidates_rank_by_accuracy_then_worst_subject():
    winner = select_phase_d_winner([baseline, candidate_a, candidate_b])
    assert winner.name == "candidate_b"
```

- [ ] **Step 2: Run RED test and implement result contracts/selection**

Eligibility values are computed from D0 frame metrics, not hard-coded window metrics.

- [ ] **Step 3: Add RED outer-boundary audit test**

Every fold result must record outer train ids, held-out id, inner splits, selected config, cache signatures, and seed. The test fails if held-out ids occur in any fit list.

- [ ] **Step 4: Implement D0 and D1 outer folds**

Generate continuous held-out frame probabilities, legacy windows, metrics, classification reports, confusion matrices, and configuration audit records.

- [ ] **Step 5: Implement D2 scale fusion**

Select scale settings and fusion weights inside inner LOSO, then evaluate once on the outer subject.

- [ ] **Step 6: Implement D3 flat/gated/decoded ablations**

Select alpha and transition strength inside inner LOSO. Save flat, gated, and decoded predictions separately.

- [ ] **Step 7: Verify Task 4**

Run synthetic four-subject end-to-end tests, then the full suite.

### Task 5: Shallow TCN Challenger

**Files:**
- Create: `phase_d_tcn.py`
- Create: `tests/test_phase_d_tcn.py`

**Interfaces:**
- Consumes: normalized fixed-length joint/bone/velocity tensors and outer/inner subject splits.
- Produces: `TCNConfig`, `MultiStreamTCN`, `fit_tcn(...)`, `predict_tcn_probabilities(...)`, `run_d4_fold(...)`.

- [ ] **Step 1: Check runtime before production edits**

Run: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

If PyTorch is unavailable, record D4 as unavailable and continue with D0-D3; do not install an unapproved GPU stack.

- [ ] **Step 2: Write RED shape and determinism tests**

Assert output shape `(batch, 8)`, finite logits, left-right augmentation mapping, and reproducible initialization with seed 42.

- [ ] **Step 3: Implement the minimal multi-stream TCN**

Implement coordinate, bone, and velocity embeddings, residual dilated temporal blocks, global pooling, multiclass head, and optional binary auxiliary head.

- [ ] **Step 4: Add RED subject-split and early-stopping tests**

Assert no random-window validation split and no outer held-out samples in augmentation, training, or stopping.

- [ ] **Step 5: Implement six fixed candidates and D4 fold adapter**

Use the shared frame aggregator and D3 decoder; serialize state dict, config, class order, normalization, and window metadata.

- [ ] **Step 6: Verify Task 5**

Run CPU smoke training and full tests before any four-fold run.

### Task 6: English Notebook, Execution, and Locked Artifacts

**Files:**
- Create: `build_phase_d_notebook.py`
- Generate: `ISAS_CHALLENGE_v11_PHASE_D_ACCURACY_ENGLISH.ipynb`
- Generate: `ISAS_CHALLENGE_v11_PHASE_D_ACCURACY_ENGLISH_EXECUTED.ipynb`
- Create: `artifacts/phase_d/*`
- Create: `outputs/phase_d/*`

**Interfaces:**
- Consumes: Tasks 1-5 public APIs and frozen data paths.
- Produces: executed English notebook, audit tables, winner artifact, secondary S4 predictions, five-subject LOSO, final all-five artifact.

- [ ] **Step 1: Generate notebook with explicit checkpoints**

Create separate outer-fold cells for S1, S2, S3, and S5 for D0-D4. Include self-critique cells after each stage showing gains, regressions, eligibility, and the decision to continue or stop.

- [ ] **Step 2: Validate before execution**

Compile every code cell, assert ASCII-only source, assert no S4 label read occurs before winner lock, and assert exact fold-cell counts.

- [ ] **Step 3: Execute D0, review, then D1-D3 sequentially**

After each stage, verify leakage audit, compare subject deltas, and stop a candidate when it violates the declared rule. Reuse signed caches.

- [ ] **Step 4: Execute D4 only after CPU/GPU smoke verification**

Record runtime and stop if the fixed candidate budget cannot be completed reliably.

- [ ] **Step 5: Lock winner and perform secondary S4 analysis**

Write predicted CSV, submission CSV, frame metrics, classification report, confusion matrix, and top confusions. Mark S4 as non-blind.

- [ ] **Step 6: Run locked five-subject LOSO and final all-five fit**

Do not reopen architecture or tuning decisions after S4.

- [ ] **Step 7: Final verification**

Require zero executed-notebook errors, loadable artifacts, 117,921 S4 prediction rows, valid eight-class labels, full test pass, and consistent class/feature schemas.
