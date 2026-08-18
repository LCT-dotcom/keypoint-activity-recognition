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
from sklearn.metrics import classification_report, confusion_matrix

from abc_experiment_pipeline import extract_experiment_unlabeled_windows, get_experiment
from phase_d_classical import TrainingOnlyFeatureSelector, fit_classical_model, phase_d_parameter_candidates
from phase_d_evaluation import aggregate_window_probabilities, estimate_transition_model, viterbi_decode
from tsfel_histgb_pipeline import (
    CLASSES,
    WindowFeatures,
    clean_labels,
    concatenate_window_features,
    evaluate_predictions,
)


SUBJECTS = (1, 2, 3, 5)


def select_global_d1_candidate(scores: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    summary = scores.groupby("candidate_index", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        worst_accuracy=("accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_abnormal_f1=("abnormal_f1", "mean"),
    )
    ranked = summary.sort_values(
        ["mean_accuracy", "worst_accuracy", "mean_macro_f1", "mean_abnormal_f1"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    return int(ranked.iloc[0]["candidate_index"]), ranked


def load_or_extract_s4_windows(
    shared_path: Path, cache_path: Path, stride: int = 75
) -> WindowFeatures:
    stat = shared_path.stat()
    signature = {
        "source_path": str(shared_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "experiment": "C",
        "window_size": 150,
        "stride": stride,
    }
    if cache_path.exists():
        try:
            payload = joblib.load(cache_path)
            if payload.get("signature") == signature:
                return payload["windows"]
        except (EOFError, OSError, ValueError):
            pass
    frame = pd.read_csv(shared_path)
    windows = extract_experiment_unlabeled_windows(
        frame, subject_id=4, experiment="C", window_size=150, stride=stride
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".joblib.tmp")
    joblib.dump({"signature": signature, "windows": windows}, temporary, compress=3)
    temporary.replace(cache_path)
    return windows


def lock_and_predict(
    data_dir: Path,
    shared_path: Path,
    c_cache_dir: Path,
    phase_dir: Path,
    output_dir: Path,
    random_state: int = 42,
) -> dict:
    windows = {
        subject: joblib.load(c_cache_dir / f"abc_c_subject_{subject}_windows.joblib")["windows"]
        for subject in SUBJECTS
    }
    truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }
    comparison = json.loads(
        (phase_dir / "final_comparison.json").read_text(encoding="utf-8")
    )
    if comparison["winner"] != "D3u-D1u-soft-gate-viterbi":
        raise ValueError("The S4 lock runner only supports the selected D3u winner")
    scores = pd.read_csv(
        phase_dir / "d1_pair_search" / "pair_search_scores_unlabeled_inference.csv"
    )
    selected_index, ranked = select_global_d1_candidate(scores)
    config = phase_d_parameter_candidates(random_state)[selected_index]
    train = concatenate_window_features([windows[subject] for subject in SUBJECTS])
    selector_path = phase_dir / "locked_phase_d_selector_subjects_1_2_3_5.joblib"
    selector_signature = {
        "subjects": list(SUBJECTS),
        "selected_candidate_index": selected_index,
        "config": config.to_dict(),
        "rows": len(train.x),
        "columns": len(train.x.columns),
    }
    selector = None
    if selector_path.exists():
        try:
            selector_payload = joblib.load(selector_path)
            if selector_payload.get("signature") == selector_signature:
                selector = selector_payload.get("selector")
        except (EOFError, OSError, ValueError):
            selector = None
    if selector is None:
        selector = TrainingOnlyFeatureSelector(
            feature_budget=config.feature_budget, random_state=random_state
        ).fit(train.x, train.y)
        joblib.dump(
            {"signature": selector_signature, "selector": selector},
            selector_path,
            compress=3,
        )
    artifact_path = phase_dir / "locked_phase_d_subjects_1_2_3_5.joblib"
    previous = joblib.load(artifact_path) if artifact_path.exists() else None
    if (
        previous
        and previous.get("selected_candidate_index") == selected_index
        and previous.get("config") == config.to_dict()
    ):
        model = previous["model"]
    else:
        model = fit_classical_model(
            train,
            config=config,
            random_state=random_state,
            prefit_selector=selector,
        )
    d3_summary = json.loads((phase_dir / "d3u_summary.json").read_text(encoding="utf-8"))
    alphas = [
        float(record["alpha"]) for record in d3_summary["outer_selections"].values()
    ]
    if any(alpha != 0.0 for alpha in alphas):
        raise ValueError("Locked D3 expected every outer fold to disable the binary gate")
    strengths = [
        float(record["transition_strength"])
        for record in d3_summary["outer_selections"].values()
    ]
    transition_strength = float(np.median(strengths))
    initial, transition = estimate_transition_model(
        [truth[subject] for subject in SUBJECTS], CLASSES, laplace=1.0
    )
    experiment = get_experiment("C")
    artifact = {
        "format_version": 2,
        "architecture": "TSFEL-C + training-only selection + HistGradientBoosting + Viterbi",
        "locked_candidate": "D3u-D1u-soft-gate-viterbi",
        "development_subjects": list(SUBJECTS),
        "selected_candidate_index": selected_index,
        "config": config.to_dict(),
        "model": model,
        "classes": tuple(CLASSES),
        "feature_schema_version": experiment.feature_schema_version,
        "preprocessing": {
            "experiment": "C",
            "window_size": 150,
            "training_stride": 75,
            "inference_stride": 75,
            "majority_threshold": 0.70,
        },
        "decoder": {
            "alpha": 0.0,
            "transition_strength": transition_strength,
            "initial_probabilities": initial,
            "transition_probabilities": transition,
        },
        "selection": {
            "source": "leakage-free S1/S2/S3/S5 nested LOSO only",
            "ranking": ranked.to_dict(orient="records"),
            "outer_strengths": strengths,
        },
    }
    joblib.dump(artifact, artifact_path, compress=3)

    shared = pd.read_csv(shared_path)
    test = load_or_extract_s4_windows(
        shared_path, phase_dir / "s4_unlabeled_c_stride75.joblib", stride=75
    )
    window_probabilities = model.predict_proba(test.x, CLASSES)
    frame_result = aggregate_window_probabilities(
        test.meta,
        window_probabilities,
        n_frames=len(shared),
        classes=CLASSES,
    )
    decoded = viterbi_decode(
        frame_result.probabilities,
        transition,
        initial,
        strength=transition_strength,
    )
    labels = np.asarray(CLASSES, dtype=object)[decoded]
    output_dir.mkdir(parents=True, exist_ok=True)
    filled = shared.copy()
    filled["Activity Label"] = labels
    filled_path = output_dir / "test_data_keypoint_shared_phase_d_predicted.csv"
    filled.to_csv(filled_path, index=False)
    submission = pd.DataFrame(
        {
            "participant_id": 4,
            "timestamp": shared["frame_id"],
            "predicted_label": labels,
        }
    )
    submission_path = output_dir / "submission_phase_d.csv"
    submission.to_csv(submission_path, index=False)
    np.savez_compressed(
        phase_dir / "locked_s4_frame_probabilities.npz",
        probabilities=frame_result.probabilities,
        coverage=frame_result.coverage,
        decoded=decoded,
    )
    manifest = {
        "artifact": str(artifact_path.resolve()),
        "filled_csv": str(filled_path.resolve()),
        "submission_csv": str(submission_path.resolve()),
        "rows": len(shared),
        "selected_candidate_index": selected_index,
        "config": config.to_dict(),
        "transition_strength": transition_strength,
        "s4_labels_used_for_lock_or_prediction": False,
    }
    (phase_dir / "locked_s4_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def evaluate_s4(label_path: Path, phase_dir: Path, output_dir: Path) -> dict:
    prediction_manifest = json.loads(
        (phase_dir / "locked_s4_prediction_manifest.json").read_text(encoding="utf-8")
    )
    predicted = pd.read_csv(prediction_manifest["filled_csv"])["Activity Label"].to_numpy(
        dtype=object
    )
    truth = clean_labels(pd.read_csv(label_path)["Action Label"]).to_numpy(dtype=object)
    if len(predicted) != len(truth):
        raise ValueError("S4 prediction and label files have different row counts")
    valid = np.asarray([label in set(CLASSES) for label in truth], dtype=bool)
    metrics = evaluate_predictions(truth[valid], predicted[valid])
    confusion = pd.DataFrame(
        confusion_matrix(truth[valid], predicted[valid], labels=CLASSES),
        index=CLASSES,
        columns=CLASSES,
    )
    report = pd.DataFrame(
        classification_report(
            truth[valid], predicted[valid], labels=CLASSES, output_dict=True, zero_division=0
        )
    ).transpose()
    confusion.to_csv(phase_dir / "locked_phase_d_s4_confusion.csv")
    report.to_csv(phase_dir / "locked_phase_d_s4_classification_report.csv")
    normalized = confusion.div(confusion.sum(axis=1), axis=0).fillna(0.0)
    misses = [
        {
            "true_label": true_label,
            "predicted_label": predicted_label,
            "row_error_rate": float(normalized.loc[true_label, predicted_label]),
            "frames": int(confusion.loc[true_label, predicted_label]),
        }
        for true_label in CLASSES
        for predicted_label in CLASSES
        if true_label != predicted_label
    ]
    top_misses = pd.DataFrame(misses).sort_values(
        ["row_error_rate", "frames"], ascending=False
    )
    top_misses.to_csv(phase_dir / "locked_phase_d_s4_top_confusions.csv", index=False)
    result = {
        **metrics,
        "error_rate": 1.0 - metrics["accuracy"],
        "evaluated_frames": int(valid.sum()),
        "non_target_frames_excluded": int((~valid).sum()),
        "s4_status": "secondary_non_blind_evaluation",
    }
    (phase_dir / "locked_phase_d_s4_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return {"metrics": result, "top_confusions": top_misses.head(10).to_dict(orient="records")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock Phase D, predict S4, then evaluate separately.")
    parser.add_argument("mode", choices=("lock-predict", "evaluate"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--shared-path", type=Path)
    parser.add_argument("--label-path", type=Path)
    parser.add_argument("--c-cache-dir", type=Path, default=Path("artifacts/abc_ablation/cache"))
    parser.add_argument("--phase-dir", type=Path, default=Path("artifacts/phase_d"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase_d"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "lock-predict":
        if args.data_dir is None or args.shared_path is None:
            raise SystemExit("lock-predict requires --data-dir and --shared-path")
        result = lock_and_predict(
            args.data_dir,
            args.shared_path,
            args.c_cache_dir,
            args.phase_dir,
            args.output_dir,
        )
    else:
        if args.label_path is None:
            raise SystemExit("evaluate requires --label-path")
        result = evaluate_s4(args.label_path, args.phase_dir, args.output_dir)
    print(json.dumps(result, indent=2))
