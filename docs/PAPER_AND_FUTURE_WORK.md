# Paper Alignment and Future Work

## Relationship to the Published Paper

The paper defines the research foundation used by this repository:

- 17 COCO 2D keypoints rather than identifiable RGB video;
- participant-local cleaning and torso-based normalization;
- five-second windows at 30 FPS;
- TSFEL statistical, temporal, and spectral descriptors;
- handcrafted position, velocity, acceleration, angle, and distance features;
- HistGradientBoosting as the strongest tested classical classifier;
- Leave-One-Subject-Out evaluation for cross-person generalization.

The Phase D implementation keeps that structure. It adds training-only feature selection, stronger regularization, nested subject-level model selection, continuous label-free held-out window extraction, and a transition model selected only from development subjects.

## Why the Reported Numbers Differ

The paper reports approximately 76% average accuracy using its original window-level evaluation. The corrected Phase D development result is 58.38% pooled frame-level accuracy on S1, S2, S3, and S5. These numbers must not be subtracted from one another: they use different prediction granularity and held-out inference construction.

The key protocol correction is that validation and held-out windows are now created at a fixed stride without reading labels. Labels are used only after prediction to calculate metrics. The earlier implementation used majority-label filtering to decide which held-out windows existed, which can make an evaluation optimistic and does not match deployment on an unlabeled stream.

## Measured Phase D Results

| Evaluation | Accuracy | Macro F1 | Abnormal F1 | Interpretation |
|---|---:|---:|---:|---|
| D0u V7-compatible baseline, S1/S2/S3/S5 | 56.63% | 55.92% | 70.01% | Continuous label-free held-out inference |
| D1u regularized HistGradientBoosting | 58.26% | 56.70% | 69.65% | Eligible under the abnormal/hard-subject guardrail |
| D3u D1u plus nested Viterbi | **58.38%** | **56.84%** | **69.71%** | Locked development winner |
| S4 secondary evaluation | **79.92%** | **84.10%** | **86.79%** | S4 excluded from model selection, but labels are now known |
| Five-subject LOSO | **73.31%** | **76.39%** | **89.34%** | Each fold trains on four of five subjects |

During four-subject development, S3 remains the hardest held-out participant at 43.79%. In five-subject LOSO, S1 becomes the lowest fold at 57.79% because every fold now trains on four participants, including S4 where applicable. These are different training sets, so the ordering is not contradictory.

## What Was Tried and Rejected

- **D2 compact multi-scale geometry:** rejected because pooled accuracy fell to 55.05%, abnormal F1 declined, and S3 dropped to 36.97%.
- **D4 shallow multi-stream TCN:** rejected because pooled accuracy fell to 51.10% and abnormal F1 to 62.83%.
- **Binary normal/abnormal gate:** nested LOSO selected an alpha of zero in every outer fold. The binary gate therefore added no useful information and is disabled in the locked model.

These negative results are retained because they constrain future development: additional model complexity does not compensate for five-subject data scarcity by itself.

## Error Analysis

The dominant S4 confusion is `Sitting quietly -> Using phone` at 34% of true sitting frames. Other major confusions are `Using phone -> Eating snacks`, `Using phone -> Biting`, and the reverse hand-to-mouth confusions. This matches the paper's discussion: seated and hand-to-face actions differ mainly through subtle wrist, elbow, and face-relative motion.

## Prioritized Future Work

### 1. Unknown and Transition Rejection

Add calibrated abstention rather than forcing every frame into one of eight activities. Select thresholds inside nested LOSO using maximum probability, class margin, and temporal disagreement. Report risk-coverage curves, known-class macro F1, unknown AUROC, and false-alert rate. This directly addresses the paper's `None` and out-of-distribution limitations.

### 2. Few-Shot Subject Calibration

Evaluate a deployment protocol with a small labeled calibration set from a new participant. Compare no adaptation, probability calibration, feature normalization updates, and a shallow subject adapter. Keep calibration windows temporally separated from evaluation windows. Report accuracy as a function of calibration seconds and prioritize the worst subject rather than only pooled accuracy.

### 3. Lightweight Domain Adaptation

Test training-only CORAL or distribution alignment on compact Phase D features before attempting a large neural model. The experiment should be accepted only when nested LOSO improves both pooled accuracy and the worst held-out subject without reducing abnormal F1 beyond the current 0.5 percentage-point guardrail.

### 4. Hybrid Graph-Temporal Representation

Use a small ST-GCN or temporal transformer as a learned embedding and concatenate it with selected TSFEL features. With only five subjects, pretraining or strong regularization is required. Compare against D3u under identical label-free inference windows and include parameter count, latency, and calibration error.

### 5. View Robustness and 3D Pose

Collect multi-view or synthetic-view data and evaluate view-relative coordinates, bone vectors, and 3D pose reconstruction. The split must hold out both participant and camera view to distinguish subject generalization from viewpoint memorization.

### 6. ESP32 Deployment

The current `.joblib` model is a desktop reference and cannot run directly on ESP32 because TSFEL and scikit-learn are unavailable in firmware. Distill D3u predictions into a compact int8 temporal model using a 150-frame ring buffer, export with TFLite Micro, and verify class-by-class parity against desktop inference. Report flash size, tensor-arena memory, latency, power, and accuracy after quantization.

## Scientific Guardrails

- Never use held-out labels to create or filter inference windows.
- Select preprocessing, features, thresholds, and decoder parameters inside subject-level validation only.
- Keep S4 secondary once its labels have been inspected.
- Report pooled, per-subject, macro-F1, abnormal-F1, and confusion matrices together.
- Treat a new architecture as an improvement only after it beats D3u under the same protocol.
