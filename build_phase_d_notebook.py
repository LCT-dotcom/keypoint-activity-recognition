from pathlib import Path

import nbformat as nbf


OUTPUT = Path("ISAS_CHALLENGE_v11_PHASE_D_ACCURACY_ENGLISH.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python (base)", "language": "python", "name": "base"},
    "language_info": {"name": "python", "version": "3.13"},
}
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# Phase D: Leakage-Free Accuracy Optimization and Final Export

This notebook preserves the V7 TSFEL + HistGradientBoosting pipeline while evaluating controlled upgrades. Model selection uses nested LOSO on S1, S2, S3, and S5 only. S4 is excluded until the method is locked. Every held-out subject is inferred with continuous, fixed-stride windows created without labels."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json, sys, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd()
PHASE_DIR = ROOT / 'artifacts' / 'phase_d'
OUTPUT_DIR = ROOT / 'outputs' / 'phase_d'
DEV_SUBJECTS = [1, 2, 3, 5]
ALL_SUBJECTS = [1, 2, 3, 4, 5]
print('Python:', sys.executable)
print('Artifacts:', PHASE_DIR.resolve())"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. Protocol and Self-Critique

- Training windows may use labels for majority-label assignment and ambiguous-window rejection.
- Validation and held-out inference windows must never use labels for inclusion, exclusion, boundaries, or overlap.
- Hyperparameters, binary-gate strength, and temporal-decoder strength are selected inside nested LOSO.
- S4 labels are secondary evaluation data and never affect the four-subject lock.
- Accuracy is optimized only among candidates that preserve abnormal F1 and hard-subject performance."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """cache_audit = []
for subject in ALL_SUBJECTS:
    path = PHASE_DIR / 'c_inference_cache' / f'c_unlabeled_stride75_subject_{subject}.joblib'
    payload = joblib.load(path)
    windows = payload['windows']
    cache_audit.append({
        'subject': subject,
        'windows': len(windows.x),
        'signature_labeled': payload['signature'].get('labeled'),
        'non_null_window_labels': int(windows.y.notna().sum()),
        'feature_count': windows.x.shape[1],
    })
cache_audit = pd.DataFrame(cache_audit)
display(cache_audit)
assert cache_audit['signature_labeled'].eq(False).all()
assert cache_audit['non_null_window_labels'].eq(0).all()"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. V7-Compatible Preprocessing and Feature Stages

The retained core is pose cleaning, participant-local normalization, fixed temporal windows, TSFEL statistics, handcrafted geometry/motion features, training-only feature selection, and HistGradientBoosting. D0 is the corrected C baseline; D1 regularizes feature capacity and the classifier; D2 tests compact multi-scale geometry; D3 adds nested temporal decoding; D4 tests a shallow temporal convolutional network."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """stages = pd.DataFrame([
    ['D0u', 'V7-compatible C features', 'HistGradientBoosting baseline', 'Unlabeled stride-75'],
    ['D1u', 'C + training-only compact selection', 'Regularized HistGradientBoosting', 'Unlabeled stride-75'],
    ['D2', 'Compact geometry at 60/150/300 frames', 'Nested scale/model fusion', 'Unlabeled continuous'],
    ['D3u', 'D1u features', 'Optional soft gate + nested Viterbi', 'Unlabeled stride-75'],
    ['D4', 'Normalized pose streams', 'Shallow multi-stream TCN', 'Unlabeled continuous'],
], columns=['candidate', 'features', 'model', 'held_out_inference'])
display(stages)"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """def show_fold(prefix, subject, title):
    metrics = pd.read_csv(PHASE_DIR / f'{prefix}_fold_metrics.csv')
    row = metrics.loc[metrics['held_out_subject'].eq(subject)]
    display(row)
    confusion = pd.read_csv(PHASE_DIR / f'{prefix}_confusion_subject_{subject}.csv', index_col=0)
    plt.figure(figsize=(9, 7))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

def show_subject_progress(subject):
    rows = []
    for prefix, name in [('d0u', 'D0u'), ('d1u', 'D1u'), ('d2', 'D2'), ('d3u', 'D3u'), ('d4', 'D4')]:
        metrics = pd.read_csv(PHASE_DIR / f'{prefix}_fold_metrics.csv')
        row = metrics.loc[metrics['held_out_subject'].eq(subject)].iloc[0].to_dict()
        row['candidate'] = name
        rows.append(row)
    display(pd.DataFrame(rows).set_index('candidate'))"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 3. Corrected Baseline Before the Accuracy Upgrade"))
for subject in (1, 2, 3, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### D0u Baseline - Held-Out S{subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"show_fold('d0u', {subject}, 'D0u Confusion Matrix - Held-Out S{subject}')"
        )
    )

cells.append(nbf.v4.new_markdown_cell("## 4. Controlled Feature and Model Upgrades"))
for subject in (1, 2, 3, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### Upgrade Comparison - Held-Out S{subject}"))
    cells.append(nbf.v4.new_code_cell(f"show_subject_progress({subject})"))

cells.append(nbf.v4.new_markdown_cell("## 5. Final Development-Set Comparison and Lock"))
cells.append(
    nbf.v4.new_code_cell(
        """comparison = json.loads((PHASE_DIR / 'final_comparison.json').read_text())
comparison_table = pd.DataFrame(comparison['candidates']).T
display(comparison_table[['eligible', 'pooled_accuracy', 'pooled_macro_f1', 'pooled_abnormal_f1', 'worst_subject_accuracy', 'eligibility_reasons']])
print('Locked winner:', comparison['winner'])
display(pd.DataFrame([comparison['selection_protocol']]))"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """locked = joblib.load(PHASE_DIR / 'locked_phase_d_subjects_1_2_3_5.joblib')
display(pd.DataFrame([{
    'locked_candidate': locked['locked_candidate'],
    'development_subjects': locked['development_subjects'],
    'selected_candidate_index': locked.get('selected_candidate_index'),
    'feature_count': len(locked['model'].selector.selected_columns) if hasattr(locked['model'], 'selector') else None,
    'transition_strength': locked.get('decoder', {}).get('transition_strength', 0.0),
}]))"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 6. S4 Prediction and Secondary Evaluation"))
cells.append(
    nbf.v4.new_code_cell(
        """s4_metrics = json.loads((PHASE_DIR / 'locked_phase_d_s4_metrics.json').read_text())
display(pd.DataFrame([s4_metrics]))
display(pd.read_csv(PHASE_DIR / 'locked_phase_d_s4_top_confusions.csv').head(10))
show_fold('locked_phase_d_s4'.replace('_fold', ''), 4, 'S4 Confusion Matrix') if False else None
confusion_s4 = pd.read_csv(PHASE_DIR / 'locked_phase_d_s4_confusion.csv', index_col=0)
plt.figure(figsize=(9, 7))
sns.heatmap(confusion_s4, annot=True, fmt='d', cmap='Blues')
plt.title('Locked Model Confusion Matrix - S4')
plt.xlabel('Predicted label'); plt.ylabel('True label'); plt.tight_layout(); plt.show()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 7. Five-Subject LOSO After S4 Labels Are Available"))
for subject in (1, 2, 3, 4, 5):
    cells.append(nbf.v4.new_markdown_cell(f"### Five-Subject LOSO - Held-Out S{subject}"))
    cells.append(
        nbf.v4.new_code_cell(
            f"show_fold('five_subject', {subject}, 'Five-Subject LOSO Confusion Matrix - Held-Out S{subject}')"
        )
    )

cells.append(nbf.v4.new_markdown_cell("## 8. Final All-Five Model Artifact"))
cells.append(
    nbf.v4.new_code_cell(
        """five_summary = json.loads((PHASE_DIR / 'five_subject_summary.json').read_text())
display(pd.DataFrame([five_summary]))
final_path = PHASE_DIR / 'final_phase_d_all_five_subjects.joblib'
final_model = joblib.load(final_path)
display(pd.DataFrame([{
    'artifact': str(final_path.resolve()),
    'size_mb': final_path.stat().st_size / 1024**2,
    'format_version': final_model['format_version'],
    'architecture': final_model['architecture'],
    'training_subjects': final_model['training_subjects'],
}]))"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## Deployment Note

The `.joblib` artifact is the reproducible desktop reference model. HistGradientBoosting and TSFEL are not directly executable on an ESP32. Embedded deployment therefore requires a separate feature-compatible lightweight model or generated C inference implementation, verified against this reference artifact on identical windows."""
    )
)

nb["cells"] = cells
nbf.write(nb, OUTPUT)
print(OUTPUT.resolve())
