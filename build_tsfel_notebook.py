from pathlib import Path

import nbformat as nbf


OUTPUT = Path("ISAS_CHALLENGE_full_pipeline_v9_V7_COMPATIBLE_ENGLISH.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (base)",
        "language": "python",
        "name": "base",
    },
    "language_info": {"name": "python", "version": "3.13"},
}

cells = []
cells.append(
    nbf.v4.new_markdown_cell(
        """# ISAS Challenge: TSFEL + HistGradientBoosting

Pipeline order: validate data, engineer pose signals, extract TSFEL features, run four independent LOSO folds for S1/S2/S3/S5, fit the final model on all four labeled subjects, and predict S4."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import inspect, json, os, sys, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import confusion_matrix
import sklearn, tsfel
from tsfel_histgb_pipeline import *
from train_tsfel_histgb import load_or_extract_subject

DATA_DIR = Path(os.environ.get('KEYPOINT_DATA_DIR', 'data/keypointlabel'))
TEST_FILE = Path(os.environ.get('KEYPOINT_TEST_FILE', 'data/test_data_keypoint_shared.csv'))
S4_LABEL_FILE = Path(os.environ.get('KEYPOINT_S4_LABEL_FILE', 'data/keypointlabel/keypoints_with_labels_4.csv'))
SUBJECTS = [1, 2, 3, 5]
PARTICIPANT_ID = 4
ARTIFACT_DIR = Path('artifacts/tsfel_histgb')
CACHE_DIR = ARTIFACT_DIR / 'cache'
OUTPUT_DIR = Path('outputs/tsfel_histgb')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Python:', sys.executable)
print('TSFEL:', getattr(tsfel, '__version__', '0.2.0'), '| sklearn:', sklearn.__version__)"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 1. Schema and Label Validation"))
cells.append(
    nbf.v4.new_code_cell(
        """training_files = {s: DATA_DIR / f'keypoints_with_labels_{s}.csv' for s in SUBJECTS}
schema_rows = []
for subject, path in training_files.items():
    header = pd.read_csv(path, nrows=2)
    validate_pose_columns(header)
    assert 'Action Label' in header.columns
    schema_rows.append({'subject': subject, 'file': path.name, 'columns': len(header.columns)})
test_header = pd.read_csv(TEST_FILE, nrows=2)
validate_pose_columns(test_header)
display(pd.DataFrame(schema_rows))
print('Test columns:', len(test_header.columns), '| labeled:', 'Action Label' in test_header.columns)"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. V7-Compatible Pose Feature Engineering Before TSFEL

This stage preserves the behavior-oriented feature engineering from v7 while making it valid for LOSO. Full-subject aggregate values are replaced by frame-level signals that TSFEL summarizes inside each window. The corrected pipeline does not depend on `custom_features.py` or `tsfel_feat.json`.

- Normalization: center on the hip midpoint and scale by shoulder width, with a torso-length fallback.
- Geometry: shoulder, hip, knee, ankle, hand-to-head, hand-to-hip, and hand-to-floor distances.
- Posture: elbow, knee, hip, torso, and shoulder-tilt angles plus posture masks.
- Motion: head, wrist, ankle, knee, and center-of-mass velocity, acceleration, and jerk.
- Behavior interactions: hand-near-head, hand-to-mouth motion, low motion, static wrists, grounded wrists, and wrist-to-COM jerk ratio."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """# This is the feature-engineering function applied to each subject before TSFEL.
print(inspect.getsource(add_pose_signals))"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """preview_raw = pd.read_csv(training_files[1], nrows=900)
preview_prepared = prepare_pose_frame(preview_raw)
preview_normalized = pose_normalize(preview_prepared)
preview_enriched, signal_columns = add_pose_signals(preview_normalized)

def feature_family(name):
    if name.startswith('dist_'):
        return 'Geometry'
    if name.startswith('angle_') or name in {'knee_flex', 'knee_extend', 'elbow_flex'}:
        return 'Posture'
    if name.startswith(('vel_', 'jerk_', 'accel_', 'delta_')) or name in {'duck_proxy', 'max_frame_jerk_rw'}:
        return 'Motion'
    return 'Behavior interaction'

feature_catalog = pd.DataFrame({
    'signal': signal_columns,
    'family': [feature_family(name) for name in signal_columns],
})
print('Feature schema:', FEATURE_SCHEMA_VERSION)
display(feature_catalog.groupby('family').size().rename('signal_count').to_frame())
display(feature_catalog)
display(preview_enriched[signal_columns].head())"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 3. Built-In TSFEL Configuration"))
cells.append(
    nbf.v4.new_code_cell(
        """config = make_tsfel_config()
enabled = {domain: list(features) for domain, features in config.items()}
display(pd.DataFrame([(domain, name) for domain, names in enabled.items() for name in names], columns=['domain', 'feature']))
print('Pose signals:', len(signal_columns))
print('TSFEL functions per signal:', sum(len(v) for v in config.values()))
print('Expected final features:', len(signal_columns) * sum(len(v) for v in config.values()))"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 4. Per-Subject Feature Extraction and Versioned Cache"))
cells.append(
    nbf.v4.new_code_cell(
        """subject_windows = {}
for subject, path in training_files.items():
    subject_windows[subject] = load_or_extract_subject(path, subject, CACHE_DIR, config, force=False)

window_counts = pd.DataFrame({s: w.y.value_counts() for s, w in subject_windows.items()}).fillna(0).astype(int)
display(window_counts)
print('Feature matrix shapes:', {s: w.x.shape for s, w in subject_windows.items()})"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. Four Independent LOSO Cells

Each cell trains on the other three subjects and evaluates one held-out subject with a 150-frame stride and no overlap in the evaluation set."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """def show_loso_fold(fold):
    metric_row = {
        'held_out_subject': fold.held_out_subject,
        'n_train_windows': fold.n_train_windows,
        'n_test_windows': fold.n_test_windows,
        **fold.metrics,
    }
    display(pd.DataFrame([metric_row]))
    display(fold.classification_report)
    plt.figure(figsize=(10, 8))
    sns.heatmap(fold.confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(f'LOSO confusion matrix - held-out S{fold.held_out_subject}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()"""
    )
)

for subject in (1, 2, 3, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### LOSO S{subject}: Held-Out Subject {subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"""fold_s{subject} = run_loso_fold(subject_windows, held_out_subject={subject}, random_state=42)
fold_s{subject}.confusion.to_csv(ARTIFACT_DIR / 'loso_confusion_matrix_subject_{subject}.csv')
show_loso_fold(fold_s{subject})"""
        )
    )

cells.append(nbf.v4.new_markdown_cell("## 6. Combined LOSO Summary"))
cells.append(
    nbf.v4.new_code_cell(
        """loso = combine_loso_folds([fold_s1, fold_s2, fold_s3, fold_s5])
loso.fold_metrics.to_csv(ARTIFACT_DIR / 'loso_fold_metrics.csv', index=False)
loso.confusion.to_csv(ARTIFACT_DIR / 'loso_confusion_matrix.csv')
loso.classification_report.to_csv(ARTIFACT_DIR / 'loso_classification_report.csv')
(ARTIFACT_DIR / 'loso_summary.json').write_text(json.dumps(loso.summary, indent=2), encoding='utf-8')
display(loso.fold_metrics)
display(pd.Series(loso.summary, name='value').to_frame())"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 7. Final Training on S1, S2, S3, and S5 and Joblib Export"))
cells.append(
    nbf.v4.new_code_cell(
        """all_windows = concatenate_window_features([subject_windows[s] for s in SUBJECTS])
final_model = fit_estimator(all_windows)
artifact = build_artifact(final_model, all_windows.x.columns, metadata={
    'training_subjects': SUBJECTS,
    'training_files': {s: str(p.resolve()) for s, p in training_files.items()},
    'n_training_windows': len(all_windows.y),
    'class_window_counts': all_windows.y.value_counts().sort_index().to_dict(),
    'loso_summary': loso.summary,
})
MODEL_PATH = ARTIFACT_DIR / 'keypoint_tsfel_histgb_v7_compatible_1_2_3_5.joblib'
save_artifact(MODEL_PATH, artifact)
print(MODEL_PATH.resolve(), MODEL_PATH.stat().st_size, 'bytes')"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 8. Artifact Reload Verification"))
cells.append(
    nbf.v4.new_code_cell(
        """artifact = load_artifact(MODEL_PATH)
assert artifact['metadata']['training_subjects'] == SUBJECTS
display(pd.Series({
    'architecture': artifact['architecture'],
    'features': len(artifact['feature_columns']),
    'training_windows': artifact['metadata']['n_training_windows'],
    'pooled_loso_accuracy': artifact['metadata']['loso_summary']['pooled_accuracy'],
}))"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 9. Shared-Test Prediction for S4"))
cells.append(
    nbf.v4.new_code_cell(
        """test_frame = pd.read_csv(TEST_FILE)
prediction = predict_frame_labels(test_frame, artifact, participant_id=PARTICIPANT_ID)
paths = write_prediction_outputs(OUTPUT_DIR, test_frame, prediction, PARTICIPANT_ID)
display(pd.Series(prediction.frame_labels).value_counts().rename('frames').to_frame())
print('Filled:', paths.filled.resolve())
print('Submission:', paths.submission.resolve())"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """filled = pd.read_csv(paths.filled)
submission = pd.read_csv(paths.submission)
assert len(filled) == len(submission) == len(test_frame)
assert submission.columns.tolist() == ['participant_id', 'timestamp', 'predicted_label']
assert not submission.predicted_label.isna().any()
display(filled.head())
display(submission.head())"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 10. Validate the Organizer-Provided S4 Ground Truth"))
cells.append(
    nbf.v4.new_code_cell(
        """s4_labeled = pd.read_csv(S4_LABEL_FILE)
validate_pose_columns(s4_labeled)
assert len(s4_labeled) == len(test_frame)
assert s4_labeled['frame_id'].equals(test_frame['frame_id'])
assert s4_labeled[['frame_id', *COORD_COLUMNS]].equals(test_frame[['frame_id', *COORD_COLUMNS]])
s4_labeled['Action Label'] = clean_labels(s4_labeled['Action Label'])
print('S4 ground truth matches the shared test exactly.')
display(s4_labeled['Action Label'].value_counts(dropna=False).rename('frames').to_frame())"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 11. Evaluate the Stage-One Model on S4 Ground Truth"))
cells.append(
    nbf.v4.new_code_cell(
        """valid_s4_frames = s4_labeled['Action Label'].isin(CLASSES)
s4_frame_evaluation = evaluate_predictions(
    s4_labeled.loc[valid_s4_frames, 'Action Label'],
    prediction.frame_labels[valid_s4_frames.to_numpy()],
)
s4_frame_confusion = pd.DataFrame(
    confusion_matrix(
        s4_labeled.loc[valid_s4_frames, 'Action Label'],
        prediction.frame_labels[valid_s4_frames.to_numpy()],
        labels=CLASSES,
    ),
    index=CLASSES,
    columns=CLASSES,
)
s4_frame_metrics = {k: v for k, v in s4_frame_evaluation.items() if k != 'classification_report'}
(ARTIFACT_DIR / 's4_stage1_frame_metrics.json').write_text(json.dumps(s4_frame_metrics, indent=2), encoding='utf-8')
s4_frame_confusion.to_csv(ARTIFACT_DIR / 's4_stage1_frame_confusion_matrix.csv')
display(pd.Series(s4_frame_metrics, name='value').to_frame())
display(pd.DataFrame(s4_frame_evaluation['classification_report']).transpose())
plt.figure(figsize=(10, 8))
sns.heatmap(s4_frame_confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Stage-One Model Confusion Matrix on S4 Ground-Truth Frames')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """subject_windows[4] = load_or_extract_subject(
    S4_LABEL_FILE, 4, CACHE_DIR, config, force=False
)
s4_evaluation_windows = select_nonoverlapping_windows(subject_windows[4])
s4_window_predictions = artifact['model'].predict(s4_evaluation_windows.x)
s4_window_evaluation = evaluate_predictions(s4_evaluation_windows.y, s4_window_predictions)
s4_window_confusion = pd.DataFrame(
    confusion_matrix(s4_evaluation_windows.y, s4_window_predictions, labels=CLASSES),
    index=CLASSES,
    columns=CLASSES,
)
s4_window_metrics = {k: v for k, v in s4_window_evaluation.items() if k != 'classification_report'}
(ARTIFACT_DIR / 's4_stage1_window_metrics.json').write_text(json.dumps(s4_window_metrics, indent=2), encoding='utf-8')
s4_window_confusion.to_csv(ARTIFACT_DIR / 's4_stage1_window_confusion_matrix.csv')
display(pd.Series(s4_window_metrics, name='value').to_frame())
display(pd.DataFrame(s4_window_evaluation['classification_report']).transpose())
plt.figure(figsize=(10, 8))
sns.heatmap(s4_window_confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Stage-One Model Confusion Matrix on Non-Overlapping S4 Windows')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## 12. Final Five-Subject LOSO

The five cells below retrain the model independently. Each fold trains on four subjects and evaluates the remaining held-out subject."""
    )
)
for subject in (1, 2, 3, 4, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### Five-Subject LOSO: Held-Out S{subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"""fold5_s{subject} = run_loso_fold(subject_windows, held_out_subject={subject}, random_state=84)
fold5_s{subject}.confusion.to_csv(ARTIFACT_DIR / 'loso5_confusion_matrix_subject_{subject}.csv')
show_loso_fold(fold5_s{subject})"""
        )
    )

cells.append(nbf.v4.new_markdown_cell("## 13. Combined Five-Subject LOSO Results"))
cells.append(
    nbf.v4.new_code_cell(
        """loso5 = combine_loso_folds([fold5_s1, fold5_s2, fold5_s3, fold5_s4, fold5_s5])
loso5.fold_metrics.to_csv(ARTIFACT_DIR / 'loso5_fold_metrics.csv', index=False)
loso5.confusion.to_csv(ARTIFACT_DIR / 'loso5_confusion_matrix.csv')
loso5.classification_report.to_csv(ARTIFACT_DIR / 'loso5_classification_report.csv')
(ARTIFACT_DIR / 'loso5_summary.json').write_text(json.dumps(loso5.summary, indent=2), encoding='utf-8')
display(loso5.fold_metrics)
display(pd.Series(loso5.summary, name='value').to_frame())"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 14. Final Model Trained on All Five Labeled Subjects"))
cells.append(
    nbf.v4.new_code_cell(
        """all_five_windows = concatenate_window_features([subject_windows[s] for s in [1, 2, 3, 4, 5]])
final_model_5 = fit_estimator(all_five_windows, random_state=84)
artifact_5 = build_artifact(final_model_5, all_five_windows.x.columns, metadata={
    'training_subjects': [1, 2, 3, 4, 5],
    'training_files': {
        **{s: str(p.resolve()) for s, p in training_files.items()},
        4: str(S4_LABEL_FILE.resolve()),
    },
    'n_training_windows': len(all_five_windows.y),
    'class_window_counts': all_five_windows.y.value_counts().sort_index().to_dict(),
    'loso_summary': loso5.summary,
})
MODEL_PATH_5 = ARTIFACT_DIR / 'keypoint_tsfel_histgb_v7_compatible_1_2_3_4_5.joblib'
save_artifact(MODEL_PATH_5, artifact_5)
artifact_5 = load_artifact(MODEL_PATH_5)
display(pd.Series({
    'model_path': str(MODEL_PATH_5.resolve()),
    'artifact_bytes': MODEL_PATH_5.stat().st_size,
    'features': len(artifact_5['feature_columns']),
    'training_windows': artifact_5['metadata']['n_training_windows'],
    'pooled_loso_accuracy': artifact_5['metadata']['loso_summary']['pooled_accuracy'],
    'pooled_loso_macro_f1': artifact_5['metadata']['loso_summary']['pooled_macro_f1'],
    'pooled_loso_abnormal_f1': artifact_5['metadata']['loso_summary']['pooled_abnormal_f1'],
}))"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## 15. Predicting Another Unseen Dataset

Use the four-subject artifact for a genuinely unseen participant. Use the five-subject artifact only after S4 ground truth has been released.

Run `predict_tsfel_histgb.py --model artifacts/tsfel_histgb/keypoint_tsfel_histgb_v7_compatible_1_2_3_4_5.joblib --input <file.csv> --participant-id <id>`."""
    )
)

nb.cells = cells
nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
