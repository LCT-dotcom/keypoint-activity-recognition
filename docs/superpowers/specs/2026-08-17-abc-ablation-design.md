# Controlled A/B/C Ablation Design

## Objective

Measure whether handcrafted features and advanced preprocessing improve subject-independent activity recognition before locking the final model. S4 ground truth is excluded from all model-selection decisions.

## Experiments

- Experiment A: hip-centered, shoulder-scaled raw pose coordinates; statistical and temporal TSFEL; HistGradientBoosting.
- Experiment B: Experiment A plus leakage-safe v7 handcrafted frame signals; identical TSFEL domains, model parameters, folds, and seed.
- Experiment C: Experiment B plus robust coordinate smoothing/scale, targeted symmetric hand/posture signals, and curated spectral TSFEL.

## Evaluation Order

1. Run identical four-fold LOSO on S1, S2, S3, and S5 for A, B, and C.
2. Compare per-subject accuracy, macro-F1, abnormal F1, per-class recall, and confusion matrices.
3. Identify the hardest subject and largest class confusions after each experiment.
4. Select the final variant by pooled abnormal F1, then worst-subject accuracy, then pooled macro-F1.
5. Fit the selected variant on all S1/S2/S3/S5 windows and evaluate S4 once.
6. Add labeled S4, run five-subject LOSO with the locked variant, then fit the final all-five artifact.

## Fairness Controls

- Random state is 42 in every experiment and fold.
- Window size, strides, majority threshold, labels, and HistGradientBoosting parameters are identical.
- A and B use identical preprocessing and TSFEL configuration; only handcrafted signals differ.
- No participant-wide labeled statistic is used as a feature.
- S4 is not used to choose A, B, or C.

## Deliverables

- English-only executed A/B/C notebook.
- Per-experiment and delta comparison tables.
- Per-subject confusion matrices for each experiment.
- Locked four-subject artifact and S4 evaluation.
- Five-subject LOSO report and final all-five artifact.
- Future-work section that does not alter the measured A/B/C pipeline.
