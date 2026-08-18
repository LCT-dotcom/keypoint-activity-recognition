# Paper Correction, GitHub Publication, and Future Work Design

## Scope

Publish a reproducible research repository for the V7-compatible TSFEL + HistGradientBoosting pipeline and its leakage-free Phase D upgrade. Include an author-corrected copy of the paper with the duplicated Figure 1 segmentation box removed, but do not alter or represent the official IOP version as corrected by the publisher.

## Repository Contents

The repository contains Python source, notebook builders, source and executed notebooks, tests, compact metrics, the two final Phase D model artifacts, S4 submission output, paper documentation, and the corrected author-copy PDF. Raw participant data, extracted feature caches, temporary files, intermediate model searches, and large regenerated prediction tables are excluded.

## Paper Correction

Replace only Figure 1 on article page 3 (PDF page 4) with a clean seven-stage vector flowchart:

1. Raw 2D pose data
2. Cleaning and torso normalization
3. Five-second sliding windows
4. TSFEL and handcrafted features
5. Correlation and training-only feature selection
6. HistGradientBoosting classification
7. LOSO evaluation

Preserve all other paper pages and page geometry. Save the result as an author-corrected copy in `paper/` and document the modification.

## Research Narrative

The README and `docs/PAPER_AND_FUTURE_WORK.md` distinguish the published paper results from the new Phase D results. They explain the corrected held-out inference protocol, why Subject 3 remains difficult during four-subject development, why S1 becomes the worst fold after adding S4 to five-subject LOSO training, and why the S4 score is secondary non-blind evaluation.

## Future Work

Future work is prioritized rather than presented as implemented results:

1. Unknown/transition rejection with calibrated abstention.
2. Few-shot subject calibration and domain adaptation.
3. Hybrid TSFEL plus graph/temporal embeddings.
4. View-invariant or 3D pose features.
5. TinyML distillation for ESP32 deployment.

Each item includes a falsifiable experiment, evaluation metric, and deployment trade-off. No future-work method is claimed as complete.

## Publication

Initialize a Git repository, create a public repository under the authenticated `LCT-dotcom` account, commit the curated scope on `main`, and push. GitHub authentication must be refreshed before remote creation. The repository name defaults to `keypoint-activity-recognition-phase-d`.
