from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_phase_d_c_inference_cache import cache_path as inference_cache_path
from phase_d_classical import TrainingOnlyFeatureSelector, fit_classical_model, phase_d_parameter_candidates
from phase_d_evaluation import aggregate_window_probabilities
from phase_d_protocol import load_unlabeled_windows
from phase_d_runner import _align_sklearn_probabilities, _build_outer_result
from phase_d_search import select_outer_configs_from_pair_scores
from run_phase_d_d1 import frame_prediction_table
from tsfel_histgb_pipeline import (
    CLASSES,
    clean_labels,
    concatenate_window_features,
    evaluate_predictions,
    fit_estimator,
)


SUBJECTS = (1, 2, 3, 5)


def save_fold(phase_dir: Path, prefix: str, fold) -> None:
    fold.confusion.to_csv(phase_dir / f"{prefix}_confusion_subject_{fold.held_out_subject}.csv")
    fold.classification_report.to_csv(
        phase_dir / f"{prefix}_classification_report_subject_{fold.held_out_subject}.csv"
    )
    frame_prediction_table(fold.frame_result).to_csv(
        phase_dir / f"{prefix}_frame_predictions_subject_{fold.held_out_subject}.csv",
        index=False,
    )
    np.savez_compressed(
        phase_dir / f"{prefix}_frame_probabilities_subject_{fold.held_out_subject}.npz",
        probabilities=fold.frame_result.probabilities,
        coverage=fold.frame_result.coverage,
    )
    (phase_dir / f"{prefix}_audit_subject_{fold.held_out_subject}.json").write_text(
        json.dumps(fold.audit, indent=2), encoding="utf-8"
    )


def summarize_folds(phase_dir: Path, prefix: str, name: str, folds: list) -> dict:
    rows = pd.DataFrame(
        [{"held_out_subject": fold.held_out_subject, **fold.metrics} for fold in folds]
    )
    rows.to_csv(phase_dir / f"{prefix}_fold_metrics.csv", index=False)
    pooled = evaluate_predictions(
        np.concatenate([fold.y_true for fold in folds]),
        np.concatenate([fold.y_pred for fold in folds]),
    )
    summary = {
        "candidate": name,
        "pooled_accuracy": pooled["accuracy"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_abnormal_f1": pooled["abnormal_f1"],
        "worst_subject_accuracy": float(rows["accuracy"].min()),
        "worst_subject": int(rows.loc[rows["accuracy"].idxmin(), "held_out_subject"]),
        "held_out_windowing": "unlabeled_continuous_stride75",
    }
    (phase_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run(data_dir: Path, labeled_cache_dir: Path, phase_dir: Path, random_state: int = 42) -> dict:
    labeled = {
        subject: joblib.load(labeled_cache_dir / f"abc_c_subject_{subject}_windows.joblib")["windows"]
        for subject in SUBJECTS
    }
    inference = {
        subject: load_unlabeled_windows(
            inference_cache_path(phase_dir / "c_inference_cache", subject)
        )
        for subject in SUBJECTS
    }
    truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }

    d0_summary_path = phase_dir / "d0u_summary.json"
    d0_metrics_path = phase_dir / "d0u_fold_metrics.csv"
    if d0_summary_path.exists() and d0_metrics_path.exists():
        d0_summary = json.loads(d0_summary_path.read_text(encoding="utf-8"))
    else:
        d0_folds = []
        for held_out in SUBJECTS:
            train_subjects = tuple(subject for subject in SUBJECTS if subject != held_out)
            train = concatenate_window_features([labeled[subject] for subject in train_subjects])
            model = fit_estimator(train, random_state=random_state + held_out)
            test = inference[held_out]
            probabilities = _align_sklearn_probabilities(model, test.x)
            frame_result = aggregate_window_probabilities(
                test.meta, probabilities, len(truth[held_out]), CLASSES
            )
            audit = {
                "outer_held_out_subject": held_out,
                "outer_train_subjects": list(train_subjects),
                "inner_splits": [],
                "candidate": "D0u-C-unlabeled-inference",
                "held_out_labels_used_for_windowing": False,
            }
            fold = _build_outer_result(
                "D0u-C-unlabeled-inference",
                held_out,
                frame_result,
                truth[held_out],
                audit,
                model,
            )
            save_fold(phase_dir, "d0u", fold)
            d0_folds.append(fold)
        d0_summary = summarize_folds(
            phase_dir, "d0u", "D0u-C-unlabeled-inference", d0_folds
        )

    candidates = phase_d_parameter_candidates(random_state)
    search_dir = phase_dir / "d1_pair_search"
    unbiased_score_path = search_dir / "pair_search_scores_unlabeled_inference.csv"
    if unbiased_score_path.exists():
        unbiased_scores = pd.read_csv(unbiased_score_path)
    else:
        score_rows = []
        for train_subjects in combinations(SUBJECTS, 2):
            validation_subjects = [
                subject for subject in SUBJECTS if subject not in train_subjects
            ]
            pair_code = "_".join(map(str, train_subjects))
            for candidate_index in range(len(candidates)):
                payload = joblib.load(
                    search_dir / f"model_pair_{pair_code}_candidate_{candidate_index:02d}.joblib"
                )
                model = payload["model"]
                for validation_subject in validation_subjects:
                    test = inference[validation_subject]
                    probabilities = model.predict_proba(test.x, CLASSES)
                    frame_result = aggregate_window_probabilities(
                        test.meta,
                        probabilities,
                        len(truth[validation_subject]),
                        CLASSES,
                    )
                    valid = np.asarray(
                        [label in set(CLASSES) for label in truth[validation_subject]],
                        dtype=bool,
                    )
                    metrics = evaluate_predictions(
                        truth[validation_subject][valid], frame_result.labels[valid]
                    )
                    score_rows.append(
                        {
                            "train_subjects": pair_code,
                            "candidate_index": candidate_index,
                            "validation_subject": validation_subject,
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "abnormal_f1": metrics["abnormal_f1"],
                        }
                    )
        unbiased_scores = pd.DataFrame(score_rows)
        unbiased_scores.to_csv(unbiased_score_path, index=False)
    selections, selection_audit = select_outer_configs_from_pair_scores(
        unbiased_scores, SUBJECTS, candidates
    )

    d1_folds = []
    for held_out in SUBJECTS:
        selection = selections[held_out]
        train_subjects = tuple(subject for subject in SUBJECTS if subject != held_out)
        legacy_audit = json.loads(
            (phase_dir / f"d1_audit_subject_{held_out}.json").read_text(encoding="utf-8")
        )
        compatible = (
            int(legacy_audit["selected_candidate_index"])
            == int(selection.audit["selected_candidate_index"])
            and legacy_audit["selected_config"] == selection.config.to_dict()
            and tuple(legacy_audit["outer_train_subjects"]) == train_subjects
        )
        if compatible:
            model = joblib.load(phase_dir / f"d1_model_outer_subject_{held_out}.joblib")
        else:
            train = concatenate_window_features([labeled[subject] for subject in train_subjects])
            selector = TrainingOnlyFeatureSelector(
                feature_budget=selection.config.feature_budget,
                random_state=random_state + held_out,
            ).fit(train.x, train.y)
            model = fit_classical_model(
                train,
                selection.config,
                random_state=random_state + held_out,
                prefit_selector=selector,
            )
        test = inference[held_out]
        probabilities = model.predict_proba(test.x, CLASSES)
        frame_result = aggregate_window_probabilities(
            test.meta, probabilities, len(truth[held_out]), CLASSES
        )
        audit = dict(selection.audit)
        audit.update(
            {
                "candidate": "D1u-regularized-C-unlabeled-inference",
                "selected_config": selection.config.to_dict(),
                "held_out_labels_used_for_windowing": False,
            }
        )
        fold = _build_outer_result(
            "D1u-regularized-C-unlabeled-inference",
            held_out,
            frame_result,
            truth[held_out],
            audit,
            model,
        )
        joblib.dump(
            model, phase_dir / f"d1u_model_outer_subject_{held_out}.joblib", compress=3
        )
        selection.inner_metrics.to_csv(
            phase_dir / f"d1u_inner_metrics_outer_subject_{held_out}.csv", index=False
        )
        save_fold(phase_dir, "d1u", fold)
        d1_folds.append(fold)
    d1_summary = summarize_folds(
        phase_dir, "d1u", "D1u-regularized-C-unlabeled-inference", d1_folds
    )
    result = {
        "d0u": d0_summary,
        "d1u": d1_summary,
        "selection_audit": selection_audit,
        "score_rows": len(unbiased_scores),
    }
    (phase_dir / "unbiased_d01_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct D0/D1 with unlabeled held-out windowing.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labeled-cache-dir", type=Path, default=Path("artifacts/abc_ablation/cache"))
    parser.add_argument("--phase-dir", type=Path, default=Path("artifacts/phase_d"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.data_dir, args.labeled_cache_dir, args.phase_dir), indent=2))
