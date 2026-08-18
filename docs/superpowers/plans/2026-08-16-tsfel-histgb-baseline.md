# TSFEL + HistGradientBoosting Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a clean TSFEL + HistGradientBoosting LOSO baseline, export a reusable `.joblib`, and predict all rows in the shared unseen-participant CSV.

**Architecture:** A reusable Python module owns label cleaning, pose normalization, derived time-series signals, TSFEL extraction, LOSO evaluation, artifact persistence, and frame-level prediction. Thin CLI scripts and a clean notebook call that module. Built-in TSFEL statistical/temporal features replace the missing external custom-feature files.

**Tech Stack:** Python 3.13 (`python`), pandas, NumPy, TSFEL 0.2.0, scikit-learn 1.7.2, SciPy, joblib, matplotlib/seaborn, pytest, nbformat.

## Global Constraints

- Training participants are exactly `1, 2, 3, 5`.
- Shared test input is `data/test_data_keypoint_shared.csv`.
- Predict exactly eight activity classes; map `Throwing` to `Throwing things`; never train a `None` class.
- Use 30 FPS, 150-frame windows, 50% training overlap, 0% held-out/test overlap, and a 70% majority threshold.
- Never create windows across participant boundaries.
- Do not require `custom_features.py` or `tsfel_feat.json`.
- Do not overwrite the original notebook or source CSV files.
- The final model artifact is `artifacts/tsfel_histgb/keypoint_tsfel_histgb_1_2_3_5.joblib`.
- The filled test output preserves all source columns and adds `predicted_label` and `prediction_confidence`.
- The official submission output contains exactly `participant_id,timestamp,predicted_label`.
- Workspace is not a Git repository, so commit steps are replaced by explicit test and file-status checkpoints.

---

### Task 1: Preprocessing and label contract

**Files:**
- Create: `tests/test_tsfel_histgb_pipeline.py`
- Create: `tsfel_histgb_pipeline.py`

**Interfaces:**
- Produces: `CLASSES`, `ABNORMAL_CLASSES`, `COORD_COLUMNS`, `clean_labels(series)`, `validate_pose_columns(df)`, `prepare_pose_frame(df)`, `pose_normalize(df)`, `add_pose_signals(df)`, and `majority_label(labels, threshold)`.
- Consumes: pandas DataFrames using the 17 COCO joints and optional `Action Label`.

- [ ] **Step 1: Write failing preprocessing tests**

```python
def make_synthetic_pose(rows: int = 300) -> pd.DataFrame:
    t = np.arange(rows, dtype=float)
    data = {"frame_id": np.arange(rows)}
    for joint_index, joint in enumerate(JOINTS):
        data[f"{joint}_x"] = 100 + joint_index * 3 + 0.05 * t
        data[f"{joint}_y"] = 200 + joint_index * 2 + np.sin(t / 15)
    data["Action Label"] = "Walking"
    return pd.DataFrame(data)

@pytest.fixture
def synthetic_pose():
    return make_synthetic_pose(300)

def test_clean_labels_merges_throwing_and_excludes_none():
    values = pd.Series(["Throwing", "Throwing things", None, "None", "Walking"])
    assert clean_labels(values).tolist() == [
        "Throwing things", "Throwing things", "None", "None", "Walking"
    ]

def test_pose_normalize_centers_hip_midpoint(synthetic_pose):
    normalized = pose_normalize(prepare_pose_frame(synthetic_pose))
    assert np.allclose((normalized.left_hip_x + normalized.right_hip_x) / 2, 0)
    assert np.allclose((normalized.left_hip_y + normalized.right_hip_y) / 2, 0)

def test_validate_pose_columns_names_missing_column(synthetic_pose):
    with pytest.raises(ValueError, match="right_ankle_y"):
        validate_pose_columns(synthetic_pose.drop(columns=["right_ankle_y"]))
```

- [ ] **Step 2: Run the preprocessing tests and verify RED**

Run:

```powershell
& 'python' -m pytest tests/test_tsfel_histgb_pipeline.py -k "clean_labels or pose_normalize or validate_pose" -v
```

Expected: collection/import failure because `tsfel_histgb_pipeline.py` does not exist.

- [ ] **Step 3: Implement the preprocessing contract**

Implement exact constants and functions. `prepare_pose_frame` converts coordinates to numeric and interpolates within one CSV. `pose_normalize` centers on the hip midpoint and scales by torso length, falling back to shoulder width. `add_pose_signals` computes distances, joint angles, per-frame motion, and returns a stable ordered signal-column list.

```python
def clean_labels(series: pd.Series) -> pd.Series:
    labels = series.astype("string").str.strip().replace({"Throwing": "Throwing things"})
    return labels.where(labels.isin(CLASSES), "None").fillna("None")

def majority_label(labels: Iterable[str], threshold: float = 0.70) -> str | None:
    values = list(labels)
    valid = [label for label in values if label in CLASSES]
    if not valid:
        return None
    label, count = Counter(valid).most_common(1)[0]
    return label if count / len(values) >= threshold else None
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Run a file checkpoint**

```powershell
Get-Item .\tsfel_histgb_pipeline.py, .\tests\test_tsfel_histgb_pipeline.py | Select-Object FullName,Length
```

Expected: both files exist and have non-zero length.

---

### Task 2: TSFEL configuration and participant-safe windows

**Files:**
- Modify: `tests/test_tsfel_histgb_pipeline.py`
- Modify: `tsfel_histgb_pipeline.py`

**Interfaces:**
- Consumes: normalized signal DataFrames from Task 1.
- Produces: `WindowFeatures(x, y, meta)`, `make_tsfel_config()`, `window_starts(length, window_size, stride, cover_tail)`, `extract_labeled_windows(df, subject_id, config)`, and `extract_unlabeled_windows(df, subject_id, config)`.

- [ ] **Step 1: Write failing TSFEL/window tests**

```python
def test_tsfel_config_uses_only_builtin_domains():
    config = make_tsfel_config()
    assert "Custom" not in config
    assert any(config[domain] for domain in ("statistical", "temporal"))

def test_labeled_windows_apply_majority_threshold():
    synthetic_pose_300 = make_synthetic_pose(300)
    synthetic_pose_300["Action Label"] = ["Walking"] * 149 + ["None"] + ["Walking"] * 90 + ["Biting"] * 60
    windows = extract_labeled_windows(synthetic_pose_300, 1, make_tsfel_config())
    assert windows.y.tolist() == ["Walking"]

def test_unlabeled_windows_cover_tail():
    synthetic_pose_301 = make_synthetic_pose(301).drop(columns=["Action Label"])
    windows = extract_unlabeled_windows(synthetic_pose_301, 4, make_tsfel_config())
    assert windows.meta.iloc[-1].end_pos == 301
```

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
& 'python' -m pytest tests/test_tsfel_histgb_pipeline.py -k "tsfel_config or labeled_windows or cover_tail" -v
```

Expected: failures because TSFEL extraction/window APIs are not implemented.

- [ ] **Step 3: Implement curated built-in TSFEL extraction**

Build configuration from:

```python
cfg = tsfel.get_features_by_domain(["statistical", "temporal"])
```

Set all feature entries to `use="no"`, then enable a fixed whitelist including mean, median, standard deviation, variance, min, max, interquartile range, mean absolute deviation, root mean square, absolute energy, autocorrelation, zero crossing rate, area under curve, and mean absolute differences when those names exist in TSFEL 0.2.0.

Each window is passed to:

```python
tsfel.time_series_features_extractor(
    config,
    window_frame,
    fs=30,
    window_size=None,
    overlap=0,
    verbose=0,
)
```

Feature names are taken from the returned DataFrame. The extraction loop operates on one participant DataFrame and records `subject_id`, `start_pos`, `end_pos`, `frame_start`, and `frame_end` in metadata.

- [ ] **Step 4: Run the Task 2 tests and full unit suite**

```powershell
& 'python' -m pytest tests/test_tsfel_histgb_pipeline.py -v
```

Expected: all Task 1 and Task 2 tests pass with no external custom-feature file.

- [ ] **Step 5: Verify the original failure is absent**

```powershell
rg -n "add_feature_json|custom_features\.py|tsfel_feat\.json" tsfel_histgb_pipeline.py
```

Expected: no matches.

---

### Task 3: HistGradientBoosting, LOSO metrics, and joblib round-trip

**Files:**
- Modify: `tests/test_tsfel_histgb_pipeline.py`
- Modify: `tsfel_histgb_pipeline.py`
- Create: `train_tsfel_histgb.py`

**Interfaces:**
- Consumes: `WindowFeatures` from Task 2.
- Produces: `make_estimator(random_state)`, `fit_estimator(train_windows)`, `evaluate_predictions(y_true, y_pred)`, `run_loso(subject_windows)`, `build_artifact(model, feature_columns, metadata)`, `save_artifact(path, bundle)`, and `load_artifact(path)`.

- [ ] **Step 1: Write failing estimator and artifact tests**

```python
def test_joblib_round_trip_reproduces_predictions(tmp_path):
    x = pd.DataFrame({"f1": [0.0, 0.1, 1.0, 1.1], "f2": [0.0, 0.2, 1.0, 1.2]})
    tiny_window_features = WindowFeatures(
        x=x,
        y=pd.Series(["Walking", "Walking", "Biting", "Biting"]),
        meta=pd.DataFrame({"subject_id": [1, 1, 2, 2]}),
    )
    model = fit_estimator(tiny_window_features)
    before = model.predict(tiny_window_features.x)
    path = tmp_path / "model.joblib"
    save_artifact(path, build_artifact(model, tiny_window_features.x.columns, {}))
    loaded = load_artifact(path)
    after = loaded["model"].predict(tiny_window_features.x)
    assert np.array_equal(before, after)

def test_abnormal_metrics_use_four_defined_classes():
    metrics = evaluate_predictions(
        ["Walking", "Attacking", "Biting"],
        ["Walking", "Attacking", "Walking"],
    )
    assert metrics["abnormal_recall"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
& 'python' -m pytest tests/test_tsfel_histgb_pipeline.py -k "joblib or abnormal_metrics" -v
```

Expected: missing estimator/artifact API failures.

- [ ] **Step 3: Implement estimator and LOSO orchestration**

Use:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("variance", VarianceThreshold(1e-10)),
    ("classifier", HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=42,
    )),
])
```

Pass balanced sample weights using `classifier__sample_weight`. `run_loso` trains four independent folds and returns fold rows, pooled true/predicted labels, per-class report, and confusion matrix. `save_artifact` stores a plain dictionary through joblib, including version metadata and exact feature-column order.

- [ ] **Step 4: Implement `train_tsfel_histgb.py`**

The CLI accepts `--data-dir`, `--subjects`, `--test-file`, `--artifact-dir`, `--cache-dir`, `--participant-id`, and `--force-reextract`. It extracts/caches features per participant, runs LOSO, writes metrics, fits the final model, saves/reloads the artifact, and delegates shared-test prediction to Task 4 APIs.

- [ ] **Step 5: Run all tests and compilation**

```powershell
& 'python' -m pytest tests/test_tsfel_histgb_pipeline.py -v
& 'python' -m py_compile .\tsfel_histgb_pipeline.py .\train_tsfel_histgb.py
```

Expected: all tests pass and compilation exits zero.

---

### Task 4: Unseen CSV prediction and output contracts

**Files:**
- Modify: `tests/test_tsfel_histgb_pipeline.py`
- Modify: `tsfel_histgb_pipeline.py`
- Create: `predict_tsfel_histgb.py`

**Interfaces:**
- Consumes: a loaded artifact and an unlabeled pose CSV.
- Produces: `FramePrediction(frame_labels, confidence, window_predictions, window_probabilities, meta)`, `OutputPaths(filled, submission)`, `predict_frame_labels(df, artifact, participant_id)`, `write_prediction_outputs(output_dir, source_df, prediction, participant_id)`, and a CLI for arbitrary unseen CSVs.

- [ ] **Step 1: Write failing frame-coverage/output tests**

```python
@pytest.fixture
def fake_artifact():
    frame = make_synthetic_pose(300)
    frame.loc[:149, "Action Label"] = "Walking"
    frame.loc[150:, "Action Label"] = "Biting"
    windows = extract_labeled_windows(frame, 1, make_tsfel_config())
    model = fit_estimator(windows)
    return build_artifact(model, windows.x.columns, {"window_size": 150, "test_stride": 150})

@pytest.fixture
def fake_prediction():
    return FramePrediction(
        frame_labels=np.asarray(["Walking"] * 301, dtype=object),
        confidence=np.ones(301),
        window_predictions=np.asarray(["Walking", "Walking", "Walking"], dtype=object),
        window_probabilities=np.ones((3, 8)) / 8,
        meta=pd.DataFrame({"start_pos": [0, 150, 151], "end_pos": [150, 300, 301]}),
    )

def test_frame_predictions_cover_every_input_row(fake_artifact):
    synthetic_pose_301 = make_synthetic_pose(301).drop(columns=["Action Label"])
    prediction = predict_frame_labels(synthetic_pose_301, fake_artifact, participant_id=4)
    assert len(prediction.frame_labels) == 301
    assert prediction.confidence.shape == (301,)

def test_submission_has_exact_three_columns(tmp_path, fake_prediction):
    synthetic_pose_301 = make_synthetic_pose(301).drop(columns=["Action Label"])
    paths = write_prediction_outputs(tmp_path, synthetic_pose_301, fake_prediction, participant_id=4)
    submission = pd.read_csv(paths.submission)
    assert submission.columns.tolist() == ["participant_id", "timestamp", "predicted_label"]
    assert submission["timestamp"].tolist() == synthetic_pose_301.frame_id.tolist()
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
& 'python' -m pytest tests/test_tsfel_histgb_pipeline.py -k "frame_predictions or submission" -v
```

Expected: prediction/output APIs are missing.

- [ ] **Step 3: Implement probability mapping and output writers**

For each non-overlapping test window, align features to the artifact's exact feature columns, call `predict_proba`, and assign the resulting class probability vector to all original positions covered by that window. The tail-padded window covers remaining rows. Average scores where windows overlap, take argmax for `predicted_label`, and retain max probability as confidence.

Write the filled file and exact three-column submission specified by the design. Validate that all labels belong to `CLASSES`, row counts match the input, and no prediction is missing.

- [ ] **Step 4: Implement `predict_tsfel_histgb.py`**

CLI arguments:

```text
--model PATH --input PATH --output-dir PATH --participant-id VALUE
```

It loads the joblib, predicts the CSV, writes both outputs, and prints the predicted class distribution.

- [ ] **Step 5: Run unit tests and compilation**

```powershell
& 'python' -m pytest tests/test_tsfel_histgb_pipeline.py -v
& 'python' -m py_compile .\predict_tsfel_histgb.py
```

Expected: full suite passes and the CLI compiles.

---

### Task 5: Clean notebook and full-data verification

**Files:**
- Create: `ISAS_CHALLENGE_full_pipeline_v8_TSFEL_FIXED.ipynb`
- Create: `artifacts/tsfel_histgb/*`
- Create: `outputs/tsfel_histgb/*`
- Modify: `README.md`

**Interfaces:**
- Notebook consumes the three Python files from Tasks 1-4.
- Notebook produces visible data checks, LOSO results, artifact metadata, test prediction distribution, and output previews.

- [ ] **Step 1: Generate the clean notebook**

Notebook cells are ordered as:

1. Purpose and challenge rules
2. Imports, paths, `sys.executable`, and package versions
3. Training/test schema validation
4. TSFEL curated configuration preview
5. Participant feature extraction/cache
6. LOSO execution and metrics table
7. Confusion matrix and classification report
8. Final fit and `.joblib` export
9. Artifact reload verification
10. Shared-test prediction
11. Filled/submission CSV previews

No cell uses `!pip install`, `add_feature_json`, `custom_features.py`, `tsfel_feat.json`, obsolete `D:` paths, random window CV, or variables defined only by later cells.

- [ ] **Step 2: Validate notebook structure before execution**

```powershell
@'
from pathlib import Path
import nbformat
from nbformat.validator import validate
p = Path('ISAS_CHALLENGE_full_pipeline_v8_TSFEL_FIXED.ipynb')
nb = nbformat.read(p, as_version=4)
validate(nb)
assert len(nb.cells) >= 10
print('VALID', len(nb.cells))
'@ | & 'python' -
```

Expected: `VALID` with at least 10 cells and exit zero.

- [ ] **Step 3: Run the full training command**

```powershell
& 'python' .\train_tsfel_histgb.py `
  --data-dir 'data/keypointlabel' `
  --subjects 1 2 3 5 `
  --test-file 'data/test_data_keypoint_shared.csv' `
  --participant-id 4
```

Expected: four LOSO fold rows, final joblib export, artifact reload, and two shared-test output files.

- [ ] **Step 4: Verify artifacts and schemas independently**

```powershell
@'
from pathlib import Path
import joblib, pandas as pd
model = Path('artifacts/tsfel_histgb/keypoint_tsfel_histgb_1_2_3_5.joblib')
filled = Path('outputs/tsfel_histgb/test_data_keypoint_shared_predicted.csv')
submission = Path('outputs/tsfel_histgb/submission_tsfel_histgb.csv')
artifact = joblib.load(model)
filled_df = pd.read_csv(filled)
submission_df = pd.read_csv(submission)
assert artifact['classes'] == [
    'Attacking', 'Biting', 'Eating snacks', 'Head banging',
    'Sitting quietly', 'Throwing things', 'Using phone', 'Walking'
]
assert len(filled_df) == len(submission_df) > 0
assert {'predicted_label', 'prediction_confidence'} <= set(filled_df.columns)
assert submission_df.columns.tolist() == ['participant_id', 'timestamp', 'predicted_label']
assert not filled_df['predicted_label'].isna().any()
print('VERIFIED', len(filled_df), model.stat().st_size)
'@ | & 'python' -
```

Expected: `VERIFIED <row_count> <artifact_size>` and exit zero.

- [ ] **Step 5: Update usage documentation and final test checkpoint**

Document notebook execution, CLI retraining, CLI inference, artifact contents, output paths, and the distinction between LOSO accuracy and unlabeled shared-test predictions. Then run:

```powershell
& 'python' -m pytest -v
& 'python' -m py_compile .\tsfel_histgb_pipeline.py .\train_tsfel_histgb.py .\predict_tsfel_histgb.py
```

Expected: all tests pass; all files compile; no source or notebook references missing custom TSFEL files.
