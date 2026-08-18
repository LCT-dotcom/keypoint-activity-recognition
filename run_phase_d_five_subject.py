from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phase_d_classical import ClassicalConfig, TrainingOnlyFeatureSelector, fit_classical_model
from phase_d_evaluation import FrameProbabilityResult, aggregate_window_probabilities, estimate_transition_model, viterbi_decode
from phase_d_protocol import load_unlabeled_windows
from phase_d_runner import _build_outer_result
from run_phase_d_d1 import frame_prediction_table
from tsfel_histgb_pipeline import CLASSES, clean_labels, concatenate_window_features, evaluate_predictions


ALL_SUBJECTS = (1, 2, 3, 4, 5)


def _selector_signature(subjects: tuple[int, ...], config: ClassicalConfig, train) -> dict:
    return {
        "subjects": list(subjects),
        "config": config.to_dict(),
        "rows": len(train.x),
        "columns": len(train.x.columns),
    }


def load_or_fit_selector(
    destination: Path,
    subjects: tuple[int, ...],
    train,
    config: ClassicalConfig,
    random_state: int,
):
    signature = _selector_signature(subjects, config, train)
    if destination.exists():
        try:
            payload = joblib.load(destination)
            if payload.get("signature") == signature:
                return payload["selector"]
        except (EOFError, OSError, ValueError):
            pass
    selector = TrainingOnlyFeatureSelector(
        feature_budget=config.feature_budget, random_state=random_state
    ).fit(train.x, train.y)
    joblib.dump({"signature": signature, "selector": selector}, destination, compress=3)
    return selector


def run(
    data_dir: Path,
    s4_label_path: Path,
    cache_dir: Path,
    phase_dir: Path,
    folds: tuple[int, ...] | None = None,
    random_state: int = 42,
) -> dict:
    locked = joblib.load(phase_dir / "locked_phase_d_subjects_1_2_3_5.joblib")
    config = ClassicalConfig(**locked["config"])
    strength = float(locked["decoder"]["transition_strength"])
    training_windows = {
        subject: joblib.load(cache_dir / f"abc_c_subject_{subject}_windows.joblib")["windows"]
        for subject in ALL_SUBJECTS
    }
    inference_windows = {
        subject: load_unlabeled_windows(
            phase_dir
            / "c_inference_cache"
            / f"c_unlabeled_stride75_subject_{subject}.joblib"
        )
        for subject in ALL_SUBJECTS
    }
    truth = {
        subject: clean_labels(
            pd.read_csv(
                s4_label_path
                if subject == 4
                else data_dir / f"keypoints_with_labels_{subject}.csv"
            )["Action Label"]
        ).to_numpy(dtype=object)
        for subject in ALL_SUBJECTS
    }
    selected_folds = folds or ALL_SUBJECTS
    if not selected_folds or any(subject not in ALL_SUBJECTS for subject in selected_folds):
        raise ValueError("Five-subject fold filter contains an unknown subject")
    fold_rows: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    for held_out in selected_folds:
        train_subjects = tuple(subject for subject in ALL_SUBJECTS if subject != held_out)
        train = concatenate_window_features(
            [training_windows[subject] for subject in train_subjects]
        )
        selector = load_or_fit_selector(
            phase_dir / f"five_subject_selector_outer_{held_out}.joblib",
            train_subjects,
            train,
            config,
            random_state + held_out,
        )
        model_path = phase_dir / f"five_subject_model_outer_{held_out}.joblib"
        signature = {
            "train_subjects": list(train_subjects),
            "config": config.to_dict(),
            "strength": strength,
            "selected_columns": list(selector.selected_columns),
        }
        payload = None
        if model_path.exists():
            try:
                candidate = joblib.load(model_path)
                if candidate.get("signature") == signature:
                    payload = candidate
            except (EOFError, OSError, ValueError):
                payload = None
        if payload is None:
            model = fit_classical_model(
                train,
                config=config,
                random_state=random_state + held_out,
                prefit_selector=selector,
            )
            initial, transition = estimate_transition_model(
                [truth[subject] for subject in train_subjects], CLASSES, laplace=1.0
            )
            payload = {
                "signature": signature,
                "model": model,
                "initial_probabilities": initial,
                "transition_probabilities": transition,
            }
            temporary = model_path.with_name(f"{model_path.name}.{os.getpid()}.tmp")
            joblib.dump(payload, temporary, compress=3)
            temporary.replace(model_path)
        model = payload["model"]
        test = inference_windows[held_out]
        window_probabilities = model.predict_proba(test.x, CLASSES)
        flat = aggregate_window_probabilities(
            test.meta,
            window_probabilities,
            n_frames=len(truth[held_out]),
            classes=CLASSES,
        )
        decoded = viterbi_decode(
            flat.probabilities,
            payload["transition_probabilities"],
            payload["initial_probabilities"],
            strength=strength,
        )
        labels = np.asarray(CLASSES, dtype=object)[decoded]
        result = FrameProbabilityResult(
            probabilities=flat.probabilities,
            coverage=flat.coverage,
            labels=labels,
            confidence=flat.probabilities[np.arange(len(labels)), decoded],
            classes=tuple(CLASSES),
        )
        audit = {
            "outer_held_out_subject": held_out,
            "outer_train_subjects": list(train_subjects),
            "inner_splits": [],
            "candidate": "locked-D3-D1-five-subject-LOSO",
            "config": config.to_dict(),
            "transition_strength": strength,
            "method_reselected_after_s4": False,
            "held_out_labels_used_for_windowing": False,
        }
        fold = _build_outer_result(
            "locked-D3-D1-five-subject-LOSO",
            held_out,
            result,
            truth[held_out],
            audit,
            model,
        )
        fold_rows.append({"held_out_subject": held_out, **fold.metrics})
        pooled_true.append(fold.y_true)
        pooled_pred.append(fold.y_pred)
        fold.confusion.to_csv(phase_dir / f"five_subject_confusion_subject_{held_out}.csv")
        fold.classification_report.to_csv(
            phase_dir / f"five_subject_classification_report_subject_{held_out}.csv"
        )
        frame_prediction_table(result).to_csv(
            phase_dir / f"five_subject_frame_predictions_subject_{held_out}.csv", index=False
        )
        (phase_dir / f"five_subject_audit_subject_{held_out}.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print(json.dumps({"held_out": held_out, **fold.metrics}), flush=True)

    if folds is not None:
        return {"completed_folds": list(selected_folds)}

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(phase_dir / "five_subject_fold_metrics.csv", index=False)
    pooled = evaluate_predictions(np.concatenate(pooled_true), np.concatenate(pooled_pred))
    summary = {
        "method": "locked-D3-D1-five-subject-LOSO",
        "pooled_accuracy": pooled["accuracy"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_abnormal_f1": pooled["abnormal_f1"],
        "worst_subject_accuracy": float(fold_metrics["accuracy"].min()),
        "worst_subject": int(
            fold_metrics.loc[fold_metrics["accuracy"].idxmin(), "held_out_subject"]
        ),
        "config_locked_before_s4": config.to_dict(),
        "transition_strength_locked_before_s4": strength,
        "method_reselected_after_s4": False,
    }
    (phase_dir / "five_subject_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    all_train = concatenate_window_features(
        [training_windows[subject] for subject in ALL_SUBJECTS]
    )
    final_selector = load_or_fit_selector(
        phase_dir / "final_all_five_selector.joblib",
        ALL_SUBJECTS,
        all_train,
        config,
        random_state,
    )
    final_model = fit_classical_model(
        all_train,
        config=config,
        random_state=random_state,
        prefit_selector=final_selector,
    )
    initial, transition = estimate_transition_model(
        [truth[subject] for subject in ALL_SUBJECTS], CLASSES, laplace=1.0
    )
    final_artifact = {
        "format_version": 2,
        "architecture": "TSFEL-C + training-only selection + HistGradientBoosting + Viterbi",
        "locked_candidate": "D3-D1-soft-gate-viterbi",
        "training_subjects": list(ALL_SUBJECTS),
        "config": config.to_dict(),
        "model": final_model,
        "classes": tuple(CLASSES),
        "preprocessing": locked["preprocessing"],
        "decoder": {
            "alpha": 0.0,
            "transition_strength": strength,
            "initial_probabilities": initial,
            "transition_probabilities": transition,
        },
        "five_subject_loso_summary": summary,
        "method_reselected_after_s4": False,
    }
    final_path = phase_dir / "final_phase_d_all_five_subjects.joblib"
    joblib.dump(final_artifact, final_path, compress=3)
    return {"summary": summary, "final_artifact": str(final_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked five-subject LOSO and final export.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--s4-label-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/abc_ablation/cache"))
    parser.add_argument("--phase-dir", type=Path, default=Path("artifacts/phase_d"))
    parser.add_argument("--folds", help="Optional comma-separated held-out subjects.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(
        args.data_dir,
        args.s4_label_path,
        args.cache_dir,
        args.phase_dir,
        tuple(map(int, args.folds.split(","))) if args.folds else None,
    )
    print(json.dumps(result, indent=2))
