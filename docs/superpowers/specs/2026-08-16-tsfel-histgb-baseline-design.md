# TSFEL + HistGradientBoosting Baseline Design

## Goal

Replace the duplicated and broken notebook pipeline with a reproducible TSFEL + HistGradientBoosting baseline that performs LOSO on participants 1, 2, 3, and 5, exports a `.joblib` artifact, and automatically predicts every row of `E:\DATA\Keypoint\test data_keypoint_shared.csv`.

## Scope

This is a baseline repair only. It remains close to the paper and original notebook. It does not introduce TCN, LSTM, GCN, model distillation, quantization, or ESP32 deployment.

## Inputs

- Training directory: `E:\DATA\Keypoint\Train Data-20260812T151048Z-1-001\Train Data\keypointlabel`
- Training files: `keypoints_with_labels_1.csv`, `keypoints_with_labels_2.csv`, `keypoints_with_labels_3.csv`, `keypoints_with_labels_5.csv`
- Shared test file: `E:\DATA\Keypoint\test data_keypoint_shared.csv`
- Sampling rate: 30 FPS
- Window length: 150 frames (5 seconds)
- Training overlap: 50%
- LOSO/test overlap: 0%
- Window label acceptance: at least 70% of the 150 frames must share one valid target label

## Target Classes

The model predicts exactly these eight classes:

1. `Attacking`
2. `Biting`
3. `Eating snacks`
4. `Head banging`
5. `Sitting quietly`
6. `Throwing things`
7. `Using phone`
8. `Walking`

`Throwing` is mapped to `Throwing things`. Missing labels, `None`, and unknown labels are excluded from supervised windows and are never model classes.

## Architecture

### Preprocessing

Each participant is processed independently. Coordinates are converted to numeric values and interpolated within that participant. Every frame is centered at the hip midpoint and scaled by torso length, with shoulder width as a fallback for degenerate torso measurements.

Custom geometric and kinematic signals are computed directly as DataFrame columns, including wrist-to-nose distances, wrist separation, knee/ankle separation, elbow/knee/hip angles, head/wrist/ankle velocity, center-of-mass velocity, and total movement. These are ordinary time-series inputs to TSFEL; the pipeline does not use TSFEL's external `Custom` domain.

### TSFEL extraction

The pipeline uses built-in `statistical` and `temporal` domains. A curated feature configuration limits extraction to stable, interpretable features to control runtime and feature dimensionality. No `custom_features.py` or `tsfel_feat.json` is required.

Windows are created inside one participant only. No window may cross participant boundaries. Training windows use 50% overlap; held-out LOSO and shared-test windows use no overlap. The final partial test window is padded from its last observed frame so every test row receives a prediction.

### Model

The classifier is a scikit-learn `Pipeline` containing median imputation, near-zero variance removal, and `HistGradientBoostingClassifier`. Balanced per-window sample weights reduce class imbalance. All preprocessing choices and feature columns used by the final estimator are stored with the artifact.

## Evaluation

LOSO runs four folds over participants 1, 2, 3, and 5. In each fold, all windows from one participant are held out. Metrics are:

- multiclass accuracy
- macro F1
- abnormal F1, precision, and recall
- per-class classification report
- pooled confusion matrix

The abnormal classes are `Attacking`, `Biting`, `Head banging`, and `Throwing things`.

## Final Training and Artifact

After LOSO, the final model trains on all valid windows from participants 1, 2, 3, and 5. It is exported as:

`artifacts/tsfel_histgb/keypoint_tsfel_histgb_1_2_3_5.joblib`

The artifact is a dictionary containing the fitted estimator, selected input feature columns, class order, TSFEL configuration, window configuration, coordinate schema, label aliases, library versions, training participants, and LOSO summary.

## Shared Test Prediction

The test file contains 35 columns: `frame_id` plus 34 keypoint coordinates and no label. The final model predicts non-overlapping 150-frame windows. Window probabilities are mapped back to frames; the padded tail receives the last window prediction.

Two output files are produced without modifying the source CSV:

- `outputs/tsfel_histgb/test_data_keypoint_shared_predicted.csv`: original 35 columns plus `predicted_label` and `prediction_confidence`
- `outputs/tsfel_histgb/submission_tsfel_histgb.csv`: exactly `participant_id,timestamp,predicted_label`

The default `participant_id` is `4`. Because the source test file has `frame_id` rather than a timestamp column, submission `timestamp` preserves the exact `frame_id` value to maintain row alignment.

## Deliverables

- `tsfel_histgb_pipeline.py`: reusable preprocessing, extraction, evaluation, artifact, and prediction logic
- `train_tsfel_histgb.py`: command-line LOSO, final fit, export, and shared-test prediction
- `predict_tsfel_histgb.py`: load an existing `.joblib` and predict another unseen CSV
- `tests/test_tsfel_histgb_pipeline.py`: synthetic unit and artifact round-trip tests
- `ISAS_CHALLENGE_full_pipeline_v8_TSFEL_FIXED.ipynb`: clean notebook orchestrating the full workflow
- LOSO metrics, confusion matrix, final `.joblib`, filled test CSV, and submission CSV

## Error Handling

The pipeline fails early with clear messages for a missing CSV, missing coordinate column, an empty file, an incompatible artifact, no valid labeled windows, mismatched feature columns, or a selected Python environment without TSFEL. It prints `sys.executable` and dependency versions at notebook startup.

## Verification

Verification consists of:

1. A regression test that demonstrates the old external-custom-feature dependency fails when the files are absent.
2. Unit tests for label normalization, pose normalization, window majority filtering, tail padding, and frame prediction coverage.
3. A TSFEL extraction smoke test on synthetic pose data.
4. A `.joblib` save/load round-trip that reproduces predictions.
5. Notebook schema validation and compilation of all Python files.
6. A full run that produces four LOSO folds, reloads the final artifact, predicts all rows of the shared test file, and validates both output schemas.
