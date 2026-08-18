# Phase D Accuracy Optimization Design

## Objective

Improve subject-independent eight-class activity recognition beyond the frozen Experiment C baseline without tuning on S4. Accuracy is the optimization target, subject to preserving abnormal-activity performance and worst-subject robustness.

The design uses a champion-challenger process. A leakage-safe optimized classical pipeline is the champion. A shallow multi-stream temporal convolutional network is the challenger. A method is retained only when it improves the fixed outer-LOSO protocol.

## Data Boundary

- Development subjects: S1, S2, S3, and S5 only.
- Outer evaluation: four LOSO folds, each holding out one complete subject.
- Inner selection: LOSO across the three subjects available inside each outer training fold.
- S4 labels are excluded from feature selection, hyperparameter selection, architecture selection, thresholds, and temporal-decoder tuning.
- S4 is a secondary locked-model analysis because its results have already been observed. A new participant is required for a genuinely blind final estimate.
- Rows labeled `None` are not one of the eight target classes. They may provide temporal boundaries but are excluded from target-class metrics.

## Frozen Baseline

Experiment C remains unchanged and is the fallback:

- robust participant-local pose normalization;
- handcrafted symmetric hand/posture signals;
- statistical, temporal, and spectral TSFEL features;
- HistGradientBoosting with random state 42;
- pooled four-subject window accuracy 55.87%;
- pooled abnormal F1 69.67%;
- worst-subject accuracy 41.96% on S3.

All Phase D comparisons must include the frozen C predictions. No result may silently replace or retune C.

## Evaluation Protocol

### Primary frame-level evaluation

Each outer held-out subject is processed as a continuous sequence. Overlapping windows produce class probabilities. Probabilities from every window covering a frame are averaged before optional hierarchical fusion and temporal decoding. Metrics are computed on valid target-class frames.

The frozen C model is re-evaluated with the same frame-level aggregation interface so C and D differ only in the declared experiment component.

### Legacy window-level evaluation

The existing non-overlapping 150-frame LOSO metrics are retained for direct continuity with A/B/C and the original notebook. They are diagnostic and are not used to override the primary frame-level selection result.

### Metrics

Report per fold and pooled:

- multiclass accuracy;
- macro F1;
- abnormal F1, precision, and recall;
- per-class precision, recall, and F1;
- confusion matrix;
- worst-subject accuracy;
- contiguous-bout accuracy and transition-region error when bout boundaries are available.

## Selection Rule

A candidate is eligible only when all conditions hold on outer LOSO:

1. pooled abnormal F1 is no more than 0.5 percentage points below the frame-level C baseline;
2. worst-subject accuracy does not fall below the frame-level C baseline;
3. no individual subject loses more than 5 accuracy points versus C.

Eligible candidates are ranked by:

1. pooled multiclass frame accuracy;
2. worst-subject frame accuracy;
3. pooled macro F1;
4. pooled abnormal F1.

An improvement smaller than 2 accuracy points is reported as marginal. If no candidate is eligible, Experiment C remains the selected model.

## D0: Common Frame Evaluator

D0 adds no model capability. It establishes a fair frame-level evaluator for frozen C:

- configurable overlapping inference stride;
- probability accumulation and coverage counts per frame;
- deterministic handling of uncovered edge frames;
- target-label masking;
- frame, bout, and transition diagnostics;
- cache and artifact signatures that include window size, stride, and feature schema.

D0 must be completed before any accuracy claim for D1-D4.

## D1: Leakage-Safe Regularized Classical Champion

D1 tests whether C is overfit because the feature dimension is close to the number of training windows.

For every inner and outer fold, preprocessing and selection are fit on training subjects only:

1. remove non-finite and constant columns;
2. remove one of each highly correlated feature pair using an absolute training-only Spearman threshold of 0.98;
3. rank remaining features with deterministic mutual information;
4. evaluate fixed feature budgets of 128, 256, 512, and 1024;
5. tune HistGradientBoosting with `ParameterSampler(n_iter=24, random_state=42)` over learning rate `{0.03, 0.06, 0.10}`, maximum leaf nodes `{15, 31, 63}`, minimum leaf size `{20, 40, 80}`, L2 regularization `{0, 1, 5}`, feature budget `{128, 256, 512, 1024}`, and weighting `{none, square-root inverse frequency}`.

The selector, selected column order, estimator, class order, and metadata are serialized together. Inference must fail clearly when the expected schema is unavailable.

## D2: Multi-Scale Geometric Ensemble

D2 addresses the fixed five-second window limitation without tripling the full TSFEL vector.

Three independent scale models use 60, 150, and 300-frame windows. Each scale extracts a compact feature set from normalized coordinates, bones, joint angles, velocities, accelerations, and targeted hand-to-head/hip distances. Per-signal summaries are restricted to mean, standard deviation, median, interquartile range, range, slope, energy, and selected quantiles.

The scale feature matrices are not concatenated. Each model is selected independently inside inner LOSO, and frame probabilities are fused with non-negative weights that sum to one. In `(60, 150, 300)` order, the fixed fusion candidates are `(1,0,0)`, `(0,1,0)`, `(0,0,1)`, `(1/3,1/3,1/3)`, `(0.5,0.5,0)`, `(0,0.5,0.5)`, `(0.5,0,0.5)`, `(0.2,0.6,0.2)`, `(0.5,0.3,0.2)`, and `(0.2,0.3,0.5)`. This controls feature dimensionality and allows short hand motions and long postural context to contribute separately.

Training strides default to half the window length. Outer inference evaluates strides of 15 and 30 frames through inner LOSO. Majority-label thresholds of 0.70 and 0.85 are inner-selected; windows that fail the threshold are excluded from supervised training.

## D3: Soft Hierarchical Fusion and Temporal Decoding

D3 targets the observed normal/abnormal and static-hand confusions while preserving exact eight-class output.

### Soft group gate

- Train the selected flat eight-class model.
- Train a binary normal-versus-abnormal model using the same outer-training boundary.
- Adjust each flat class probability by the probability of its group raised to an inner-selected exponent from `{0, 0.5, 1, 2}`.
- Renormalize to eight-class probabilities.
- Never use a hard gate; a binary error must not make four classes unreachable.

### Temporal decoder

- Estimate an eight-state transition matrix and initial distribution from training-subject label bouts only, with Laplace smoothing.
- Decode averaged frame log probabilities with Viterbi.
- Select the transition-strength coefficient inside inner LOSO from `{0, 0.25, 0.5, 1, 2}`, so the decoder is retained only when it improves held-out subjects.
- Do not hand-code impossible transitions or activity durations from S4.

The flat, fused, and decoded results are all reported. This separates gains from classification and postprocessing.

## D4: Shallow Multi-Stream TCN Challenger

D4 is evaluated only after D1-D3 establish a strong classical champion.

### Inputs

- normalized joint coordinates;
- bone vectors and bone lengths;
- first-order joint velocities;
- optional confidence or missingness mask when available.

### Architecture

- one lightweight embedding per stream;
- three or four residual temporal convolution blocks with dilations;
- global temporal pooling;
- eight-class classification head;
- optional binary auxiliary head for normal versus abnormal activity.

The model must remain shallow. Transformer and large GCN backbones are excluded from Phase D because four development subjects do not support a reliable architecture search.

### Training controls

- deterministic seed 42 where supported;
- coordinate jitter, small rotation/scale perturbation, temporal crop/resampling, joint dropout, and anatomically correct left-right mirroring;
- augmentation applied to outer-training data only;
- early stopping selected on inner held-out subjects, never on random windows from the outer-training subjects;
- evaluate a fixed six-candidate architecture list spanning channels `{32, 64}`, block counts `{3, 4}`, dropout `{0.2, 0.4}`, cross-entropy versus focal loss, and auxiliary-head weight `{0, 0.2}`; kernel size is fixed at 5 and no unlisted architecture is added after outer results are visible.

D4 probabilities pass through the same frame aggregator and optional D3 decoder. It wins only through the common selection rule.

## Experiment Order and Stop Rules

1. Reproduce D0 frame-level C baseline.
2. Run D1. Stop feature-budget expansion when two larger budgets fail to improve inner accuracy.
3. Run D2 using the best D1 settings per scale.
4. Run D3 flat, fused, and decoded ablations.
5. Run D4 only if the required deep-learning runtime is available and validated.
6. Select one locked winner from outer LOSO without reading S4 labels during selection.
7. Fit the winner on all S1/S2/S3/S5 development data.
8. Produce a secondary S4 prediction and comparison, clearly labeled non-blind.
9. Add S4 only after the secondary analysis, run five-subject LOSO with the locked method, and export the final all-five artifact.

Computational searches are bounded and resumable. Each configuration receives a stable signature and cache path. Failed configurations are recorded with the error rather than silently omitted.

## Testing

- Unit tests for training-only feature selection and stable column ordering.
- Leakage tests that fail if an outer held-out subject influences preprocessing, ranking, tuning, fusion, or decoding.
- Synthetic tests for overlapping frame aggregation, edge coverage, soft group fusion, and Viterbi decoding.
- Artifact round-trip and schema-mismatch tests.
- Reproducibility tests for fixed seeds and configuration signatures.
- Notebook/source validation for English-only code and independent outer-fold result cells.
- End-to-end smoke test on small synthetic sequences before full extraction and training.

## Deliverables

- English Phase D source notebook and executed notebook;
- reusable Phase D experiment module and tests;
- per-configuration inner and outer LOSO tables;
- D0-D4 comparison table with eligibility decisions;
- per-subject and pooled confusion matrices;
- feature-budget and model-selection audit trail;
- locked four-subject artifact and secondary S4 output;
- final five-subject LOSO report and all-five artifact;
- concise limitations section stating that S4 is no longer a blind test.

## Explicit Non-Goals

- No S4-driven feature engineering or hyperparameter adjustment.
- No unbounded Optuna or architecture search.
- No large Transformer, language-model, or state-of-the-art GCN claim on this small dataset.
- No ESP32 conversion until the accuracy winner is locked.
- No claim that a modern architecture is better without outer-subject evidence.

## References Informing Future Extensions

- Kang et al., "Efficient Skeleton-Based Action Recognition via Joint-Mapping Strategies," WACV 2023.
- Abdelfattah et al., "MaskCLR: Attention-Guided Contrastive Learning for Robust Action Representation Learning," CVPR 2024.
- Bian et al., "Class-Aware Contrastive Learning for Fine-Grained Skeleton-Based Action Recognition," ACCV 2024.
