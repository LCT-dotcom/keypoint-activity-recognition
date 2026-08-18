# V7-Compatible English Pipeline Design

## Objective

Rebuild the TSFEL + HistGradientBoosting notebook so it follows the original v7 data flow while correcting the failures that made v7 unreliable. The notebook, code comments, printed messages, plots, and generated report labels must be English-only.

## Pipeline Contract

1. Load subjects 1, 2, 3, and 5 independently and normalize each pose sequence before windowing.
2. Map `Throwing` to `Throwing things`; treat missing/unknown/`None` rows as transitions rather than a ninth class.
3. Apply v7-style handcrafted frame-level features before TSFEL: body distances, hand-to-head proximity, joint angles, head/wrist/leg motion, jerk, center-of-mass motion, posture masks, and behavior-oriented interaction signals.
4. Extract curated built-in TSFEL statistical, temporal, and spectral features from 150-frame windows. Do not use `custom_features.py` or `tsfel_feat.json`.
5. Use 75-frame training stride, 150-frame held-out/test stride, and a 70% label-majority threshold.
6. Run four independent LOSO cells: hold out S1, S2, S3, and S5 respectively. Each cell prints its own metrics, classification report, and confusion matrix.
7. Combine the four fold outputs only after all independent cells finish.
8. Fit the final HistGradientBoosting model on all labeled windows from S1/S2/S3/S5, export/reload a `.joblib`, then predict every row of the shared S4 CSV.

## Corrections Relative to v7

- No windows may cross participant boundaries.
- No random CV may be applied to overlapping windows.
- Full-subject aggregates such as subject-wide maxima, means, percentages, and quantile thresholds are replaced by frame-level signals aggregated inside each window by TSFEL. This prevents held-out subject leakage.
- Broken external TSFEL Custom-domain files are replaced with built-in frequency-domain features.
- Cache metadata includes a feature-schema version and TSFEL configuration signature so stale v8 features cannot be reused.
- Feature columns are aligned exactly during inference and stored in the joblib artifact.

## Feature Families

- Geometry: shoulder/hip/knee/ankle/wrist distances, wrist-to-head/hip/floor distances.
- Posture: elbow, knee, hip, torso, and shoulder-tilt angles; knee-flexion/extension and elbow-flexion masks.
- Motion: head, wrist, ankle, knee, shoulder, and center-of-mass velocity; wrist/head/COM jerk and wrist acceleration.
- Behavior interactions: hand-near-head, bite proximity, micro/strong hand-to-mouth motion, static wrist, low motion, grounded wrists, duck proxy, and wrist-to-COM jerk ratio.
- TSFEL: robust statistical summaries, temporal dynamics/peaks, and spectral frequency/energy/entropy summaries.

## Outputs

- Executed English notebook with four independent LOSO result cells.
- Per-subject confusion matrix CSV files and pooled LOSO metrics.
- Final v7-compatible joblib artifact and S4 filled/submission CSV files.
- Automated tests covering feature families, spectral TSFEL configuration, cache invalidation, fold isolation, artifact round-trip, and prediction schema.
