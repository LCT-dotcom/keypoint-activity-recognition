from pathlib import Path

import nbformat as nbf


OUTPUT = Path("ISAS_CHALLENGE_v10_ABC_ABLATION_ENGLISH.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python (base)", "language": "python", "name": "base"},
    "language_info": {"name": "python", "version": "3.13"},
}
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# Controlled A/B/C Ablation for Subject-Independent Keypoint Recognition

This notebook measures improvements in a fixed order before S4 is used: baseline A, handcrafted-feature B, targeted robust C, model lock, S4 evaluation, five-subject LOSO, and final all-five training."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json, os, sys, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import confusion_matrix
import sklearn, tsfel

from abc_experiment_pipeline import *
from tsfel_histgb_pipeline import (
    CLASSES, COORD_COLUMNS, WindowFeatures, combine_loso_folds,
    concatenate_window_features, clean_labels, evaluate_predictions,
    fit_estimator, run_loso_fold, save_artifact, select_nonoverlapping_windows,
    validate_pose_columns, write_prediction_outputs,
)

DATA_DIR = Path(os.environ.get('KEYPOINT_DATA_DIR', 'data/keypointlabel'))
S4_SHARED_FILE = Path(os.environ.get('KEYPOINT_TEST_FILE', 'data/test_data_keypoint_shared.csv'))
S4_LABEL_FILE = Path(os.environ.get('KEYPOINT_S4_LABEL_FILE', 'data/keypointlabel/keypoints_with_labels_4.csv'))
SUBJECTS_STAGE1 = [1, 2, 3, 5]
ALL_SUBJECTS = [1, 2, 3, 4, 5]
RANDOM_STATE = 42
ARTIFACT_DIR = Path('artifacts/abc_ablation')
CACHE_DIR = ARTIFACT_DIR / 'cache'
OUTPUT_DIR = Path('outputs/abc_ablation')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Python:', sys.executable)
print('TSFEL:', getattr(tsfel, '__version__', '0.2.0'), '| sklearn:', sklearn.__version__)"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## Experimental Controls

- A and B use identical preprocessing, TSFEL domains, windows, folds, model parameters, and random seed.
- B differs from A only by adding handcrafted frame-level signals.
- C adds robust smoothing/scale, symmetric targeted signals, and spectral TSFEL.
- S4 labels are not loaded until the A/B/C winner has been selected.
- Participant-local deterministic preprocessing is allowed before LOSO; imputation, variance filtering, and the classifier are fitted only inside each training fold."""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 1. Data and Schema Validation"))
cells.append(
    nbf.v4.new_code_cell(
        """training_files = {s: DATA_DIR / f'keypoints_with_labels_{s}.csv' for s in SUBJECTS_STAGE1}
schema = []
for subject, path in training_files.items():
    header = pd.read_csv(path, nrows=2)
    validate_pose_columns(header)
    assert 'Action Label' in header.columns
    schema.append({'subject': subject, 'file': path.name, 'columns': len(header.columns)})
shared_header = pd.read_csv(S4_SHARED_FILE, nrows=2)
validate_pose_columns(shared_header)
display(pd.DataFrame(schema))
print('S4 shared-test columns:', len(shared_header.columns))"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 2. Experiment Definitions"))
cells.append(
    nbf.v4.new_code_cell(
        """experiment_table = pd.DataFrame([
    {
        'experiment': name,
        'schema': definition.feature_schema_version,
        'signal_mode': definition.signal_mode,
        'robust_preprocessing': definition.robust_preprocessing,
        'spectral_tsfel': definition.include_spectral,
        'random_state': definition.random_state,
    }
    for name, definition in EXPERIMENTS.items()
])
display(experiment_table)
assert experiment_table['random_state'].eq(RANDOM_STATE).all()"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """def show_fold(fold, title):
    row = {
        'held_out_subject': fold.held_out_subject,
        'n_train_windows': fold.n_train_windows,
        'n_test_windows': fold.n_test_windows,
        **fold.metrics,
    }
    display(pd.DataFrame([row]))
    display(fold.classification_report)
    plt.figure(figsize=(10, 8))
    sns.heatmap(fold.confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

def summarize_experiment(name, result):
    metrics = result.fold_metrics
    worst = metrics.loc[metrics['accuracy'].idxmin()]
    return {
        'experiment': name,
        'mean_fold_accuracy': result.summary['mean_fold_accuracy'],
        'pooled_accuracy': result.summary['pooled_accuracy'],
        'pooled_macro_f1': result.summary['pooled_macro_f1'],
        'pooled_abnormal_f1': result.summary['pooled_abnormal_f1'],
        'worst_subject': int(worst['held_out_subject']),
        'worst_subject_accuracy': float(worst['accuracy']),
    }

def top_confusions(result, limit=10):
    rows = []
    for true_index, true_label in enumerate(CLASSES):
        for pred_index, predicted_label in enumerate(CLASSES):
            count = int(result.confusion.iloc[true_index, pred_index])
            if true_index != pred_index and count:
                rows.append({'true_label': true_label, 'predicted_label': predicted_label, 'windows': count})
    return pd.DataFrame(rows).sort_values('windows', ascending=False).head(limit)"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 3. Shared Preprocessing and Experiment B Extraction"))
cells.append(
    nbf.v4.new_code_cell(
        """windows_b = {}
for subject, path in training_files.items():
    print(f'Experiment B: loading/extracting subject {subject}', flush=True)
    windows_b[subject] = load_or_extract_experiment_subject(path, subject, 'B', CACHE_DIR)
display(pd.DataFrame({s: w.y.value_counts() for s, w in windows_b.items()}).fillna(0).astype(int))
print('Experiment B shapes:', {s: w.x.shape for s, w in windows_b.items()})"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 4. Experiment A: Raw Normalized Keypoints"))
cells.append(
    nbf.v4.new_code_cell(
        """windows_a = {}
for subject, path in training_files.items():
    windows_a[subject] = derive_experiment_a_windows(windows_b[subject])
    save_experiment_cache(CACHE_DIR, path, subject, 'A', windows_a[subject])
print('Experiment A shapes:', {s: w.x.shape for s, w in windows_a.items()})"""
    )
)
for subject in (1, 2, 3, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### Experiment A - Held-Out S{subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"""a_s{subject} = run_loso_fold(windows_a, held_out_subject={subject}, random_state=RANDOM_STATE)
a_s{subject}.confusion.to_csv(ARTIFACT_DIR / 'a_confusion_subject_{subject}.csv')
show_fold(a_s{subject}, 'Experiment A Confusion Matrix - Held-Out S{subject}')"""
        )
    )
cells.append(
    nbf.v4.new_code_cell(
        """result_a = combine_loso_folds([a_s1, a_s2, a_s3, a_s5])
result_a.fold_metrics.to_csv(ARTIFACT_DIR / 'a_fold_metrics.csv', index=False)
display(result_a.fold_metrics)
display(pd.DataFrame([summarize_experiment('A', result_a)]))
display(top_confusions(result_a))"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 5. Experiment B: Add V7 Handcrafted Signals"))
for subject in (1, 2, 3, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### Experiment B - Held-Out S{subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"""b_s{subject} = run_loso_fold(windows_b, held_out_subject={subject}, random_state=RANDOM_STATE)
b_s{subject}.confusion.to_csv(ARTIFACT_DIR / 'b_confusion_subject_{subject}.csv')
show_fold(b_s{subject}, 'Experiment B Confusion Matrix - Held-Out S{subject}')"""
        )
    )
cells.append(
    nbf.v4.new_code_cell(
        """result_b = combine_loso_folds([b_s1, b_s2, b_s3, b_s5])
result_b.fold_metrics.to_csv(ARTIFACT_DIR / 'b_fold_metrics.csv', index=False)
display(result_b.fold_metrics)
display(pd.DataFrame([summarize_experiment('B', result_b)]))
display(top_confusions(result_b))

ab_subject_delta = result_a.fold_metrics[['held_out_subject', 'accuracy', 'macro_f1', 'abnormal_f1']].merge(
    result_b.fold_metrics[['held_out_subject', 'accuracy', 'macro_f1', 'abnormal_f1']],
    on='held_out_subject', suffixes=('_a', '_b')
)
for metric in ['accuracy', 'macro_f1', 'abnormal_f1']:
    ab_subject_delta[f'{metric}_delta_b_minus_a'] = ab_subject_delta[f'{metric}_b'] - ab_subject_delta[f'{metric}_a']
display(ab_subject_delta)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. Hard-Subject Analysis Before Experiment C

Experiment C is targeted at cross-subject instability and hand-activity confusion. It adds participant-local robust smoothing/scale, symmetric hand features, and spectral summaries while preserving the same labels, windows, folds, classifier, and seed."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """hard_subjects_ab = pd.DataFrame([
    summarize_experiment('A', result_a),
    summarize_experiment('B', result_b),
])
display(hard_subjects_ab[['experiment', 'worst_subject', 'worst_subject_accuracy', 'pooled_abnormal_f1', 'pooled_macro_f1']])
print('Experiment A top confusions:')
display(top_confusions(result_a))
print('Experiment B top confusions:')
display(top_confusions(result_b))"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 7. Experiment C: Targeted Robust and Spectral Upgrade"))
cells.append(
    nbf.v4.new_code_cell(
        """windows_c = {}
for subject, path in training_files.items():
    print(f'Experiment C: loading/extracting subject {subject}', flush=True)
    windows_c[subject] = load_or_extract_experiment_subject(path, subject, 'C', CACHE_DIR)
print('Experiment C shapes:', {s: w.x.shape for s, w in windows_c.items()})"""
    )
)
for subject in (1, 2, 3, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### Experiment C - Held-Out S{subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"""c_s{subject} = run_loso_fold(windows_c, held_out_subject={subject}, random_state=RANDOM_STATE)
c_s{subject}.confusion.to_csv(ARTIFACT_DIR / 'c_confusion_subject_{subject}.csv')
show_fold(c_s{subject}, 'Experiment C Confusion Matrix - Held-Out S{subject}')"""
        )
    )
cells.append(
    nbf.v4.new_code_cell(
        """result_c = combine_loso_folds([c_s1, c_s2, c_s3, c_s5])
result_c.fold_metrics.to_csv(ARTIFACT_DIR / 'c_fold_metrics.csv', index=False)
display(result_c.fold_metrics)
display(pd.DataFrame([summarize_experiment('C', result_c)]))
display(top_confusions(result_c))"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 8. Controlled A/B/C Comparison and Model Lock"))
cells.append(
    nbf.v4.new_code_cell(
        """comparison = pd.DataFrame([
    summarize_experiment('A', result_a),
    summarize_experiment('B', result_b),
    summarize_experiment('C', result_c),
])
comparison.to_csv(ARTIFACT_DIR / 'abc_summary.csv', index=False)
selected_experiment = select_winning_experiment(comparison)
(ARTIFACT_DIR / 'selected_experiment.json').write_text(
    json.dumps({'selected_experiment': selected_experiment, 'selection_order': ['pooled_abnormal_f1', 'worst_subject_accuracy', 'pooled_macro_f1']}, indent=2),
    encoding='utf-8'
)
display(comparison)
print('Locked experiment before S4 ground truth:', selected_experiment)

fold_comparison = result_a.fold_metrics[['held_out_subject', 'accuracy']].rename(columns={'accuracy': 'accuracy_a'})
fold_comparison = fold_comparison.merge(result_b.fold_metrics[['held_out_subject', 'accuracy']].rename(columns={'accuracy': 'accuracy_b'}), on='held_out_subject')
fold_comparison = fold_comparison.merge(result_c.fold_metrics[['held_out_subject', 'accuracy']].rename(columns={'accuracy': 'accuracy_c'}), on='held_out_subject')
fold_comparison['delta_b_minus_a'] = fold_comparison['accuracy_b'] - fold_comparison['accuracy_a']
fold_comparison['delta_c_minus_b'] = fold_comparison['accuracy_c'] - fold_comparison['accuracy_b']
fold_comparison.to_csv(ARTIFACT_DIR / 'abc_subject_accuracy_deltas.csv', index=False)
display(fold_comparison)"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 9. Fit the Locked Four-Subject Model"))
cells.append(
    nbf.v4.new_code_cell(
        """windows_by_experiment = {'A': windows_a, 'B': windows_b, 'C': windows_c}
results_by_experiment = {'A': result_a, 'B': result_b, 'C': result_c}
selected_windows = windows_by_experiment[selected_experiment]
selected_result = results_by_experiment[selected_experiment]
all_stage1_windows = concatenate_window_features([selected_windows[s] for s in SUBJECTS_STAGE1])
locked_model = fit_estimator(all_stage1_windows, random_state=RANDOM_STATE)
locked_artifact = build_experiment_artifact(
    locked_model,
    all_stage1_windows.x.columns,
    selected_experiment,
    metadata={
        'training_subjects': SUBJECTS_STAGE1,
        'n_training_windows': len(all_stage1_windows.y),
        'selection_summary': comparison.to_dict('records'),
        'loso_summary': selected_result.summary,
    },
)
LOCKED_MODEL_PATH = ARTIFACT_DIR / f'locked_experiment_{selected_experiment.lower()}_subjects_1_2_3_5.joblib'
save_experiment_artifact(LOCKED_MODEL_PATH, locked_artifact)
locked_artifact = load_experiment_artifact(LOCKED_MODEL_PATH)
print('Locked artifact:', LOCKED_MODEL_PATH.resolve())"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 10. S4 Prediction After Model Lock"))
cells.append(
    nbf.v4.new_code_cell(
        """s4_shared = pd.read_csv(S4_SHARED_FILE)
s4_prediction = predict_experiment_frame_labels(s4_shared, locked_artifact, participant_id=4)
s4_output_paths = write_prediction_outputs(OUTPUT_DIR, s4_shared, s4_prediction, participant_id=4)
display(pd.Series(s4_prediction.frame_labels).value_counts().rename('frames').to_frame())
print('Filled S4 CSV:', s4_output_paths.filled.resolve())
print('S4 submission CSV:', s4_output_paths.submission.resolve())"""
    )
)
cells.append(nbf.v4.new_markdown_cell("## 11. Load S4 Ground Truth and Evaluate the Locked Model"))
cells.append(
    nbf.v4.new_code_cell(
        """s4_labeled = pd.read_csv(S4_LABEL_FILE)
assert s4_labeled[['frame_id', *COORD_COLUMNS]].equals(s4_shared[['frame_id', *COORD_COLUMNS]])
s4_truth = clean_labels(s4_labeled['Action Label'])
valid_s4 = s4_truth.isin(CLASSES)
s4_evaluation = evaluate_predictions(s4_truth[valid_s4], s4_prediction.frame_labels[valid_s4.to_numpy()])
s4_metrics = {k: v for k, v in s4_evaluation.items() if k != 'classification_report'}
s4_metrics['error_rate'] = 1.0 - s4_metrics['accuracy']
(ARTIFACT_DIR / 'locked_model_s4_metrics.json').write_text(json.dumps(s4_metrics, indent=2), encoding='utf-8')
pd.DataFrame(s4_evaluation['classification_report']).transpose().to_csv(
    ARTIFACT_DIR / 'locked_model_s4_classification_report.csv'
)
s4_confusion = pd.DataFrame(
    confusion_matrix(
        s4_truth[valid_s4],
        s4_prediction.frame_labels[valid_s4.to_numpy()],
        labels=CLASSES,
    ),
    index=CLASSES,
    columns=CLASSES,
)
s4_confusion.to_csv(ARTIFACT_DIR / 'locked_model_s4_confusion.csv')
s4_confusion_pairs = []
for true_label in CLASSES:
    for predicted_label in CLASSES:
        if true_label != predicted_label and s4_confusion.loc[true_label, predicted_label] > 0:
            s4_confusion_pairs.append({
                'true_label': true_label,
                'predicted_label': predicted_label,
                'count': int(s4_confusion.loc[true_label, predicted_label]),
            })
s4_top_confusions = pd.DataFrame(s4_confusion_pairs).sort_values('count', ascending=False).head(15)
s4_top_confusions.to_csv(ARTIFACT_DIR / 'locked_model_s4_top_confusions.csv', index=False)
display(pd.Series(s4_metrics, name='value').to_frame())
display(pd.DataFrame(s4_evaluation['classification_report']).transpose())
display(s4_top_confusions)
plt.figure(figsize=(10, 8))
sns.heatmap(s4_confusion, annot=True, fmt='d', cmap='Blues')
plt.title(f'Locked experiment {selected_experiment}: S4 frame-level confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 12. Five-Subject LOSO with the Locked Experiment"))
cells.append(
    nbf.v4.new_code_cell(
        """s4_selected_windows = load_or_extract_experiment_subject(
    S4_LABEL_FILE, 4, selected_experiment, CACHE_DIR
)
five_subject_windows = {**selected_windows, 4: s4_selected_windows}
print('Five-subject feature shapes:', {s: w.x.shape for s, w in sorted(five_subject_windows.items())})"""
    )
)
for subject in (1, 2, 3, 4, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### Locked Five-Subject LOSO - Held-Out S{subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"""final_s{subject} = run_loso_fold(five_subject_windows, held_out_subject={subject}, random_state=RANDOM_STATE)
final_s{subject}.confusion.to_csv(ARTIFACT_DIR / 'final_loso5_confusion_subject_{subject}.csv')
show_fold(final_s{subject}, 'Locked Five-Subject LOSO - Held-Out S{subject}')"""
        )
    )
cells.append(
    nbf.v4.new_code_cell(
        """final_loso5 = combine_loso_folds([final_s1, final_s2, final_s3, final_s4, final_s5])
final_loso5.fold_metrics.to_csv(ARTIFACT_DIR / 'final_loso5_fold_metrics.csv', index=False)
(ARTIFACT_DIR / 'final_loso5_summary.json').write_text(json.dumps(final_loso5.summary, indent=2), encoding='utf-8')
display(final_loso5.fold_metrics)
display(pd.Series(final_loso5.summary, name='value').to_frame())"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 13. Final All-Five Model Export"))
cells.append(
    nbf.v4.new_code_cell(
        """all_five_windows = concatenate_window_features([five_subject_windows[s] for s in ALL_SUBJECTS])
final_model = fit_estimator(all_five_windows, random_state=RANDOM_STATE)
final_artifact = build_experiment_artifact(
    final_model,
    all_five_windows.x.columns,
    selected_experiment,
    metadata={
        'training_subjects': ALL_SUBJECTS,
        'n_training_windows': len(all_five_windows.y),
        'stage1_selection_summary': comparison.to_dict('records'),
        'five_subject_loso_summary': final_loso5.summary,
    },
)
FINAL_MODEL_PATH = ARTIFACT_DIR / f'final_experiment_{selected_experiment.lower()}_subjects_1_2_3_4_5.joblib'
save_experiment_artifact(FINAL_MODEL_PATH, final_artifact)
final_artifact = load_experiment_artifact(FINAL_MODEL_PATH)
display(pd.Series({
    'selected_experiment': selected_experiment,
    'artifact_path': str(FINAL_MODEL_PATH.resolve()),
    'artifact_bytes': FINAL_MODEL_PATH.stat().st_size,
    'feature_count': len(final_artifact['feature_columns']),
    'training_windows': final_artifact['metadata']['n_training_windows'],
    'five_subject_pooled_accuracy': final_loso5.summary['pooled_accuracy'],
    'five_subject_pooled_abnormal_f1': final_loso5.summary['pooled_abnormal_f1'],
}))"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## 14. Future Work (Not Included in This Ablation)

Future experiments may evaluate graph-based skeleton models, temporal convolution/transformer models, self-supervised pose pretraining, domain-adversarial learning, CORAL-style subject alignment, and confidence-aware temporal decoding. These methods must be compared against the locked A/B/C result in a separate protocol."""
    )
)

nb.cells = cells
nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
