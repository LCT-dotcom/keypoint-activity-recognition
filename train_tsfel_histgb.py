from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from tsfel_histgb_pipeline import (
    FEATURE_SCHEMA_VERSION,
    build_artifact,
    concatenate_window_features,
    extract_labeled_windows,
    fit_estimator,
    load_artifact,
    make_tsfel_config,
    predict_frame_labels,
    run_loso,
    save_artifact,
    write_prediction_outputs,
)


def build_cache_signature(csv_path: Path, config: dict) -> dict:
    config_payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return {
        "path": str(csv_path.resolve()),
        "size": csv_path.stat().st_size,
        "mtime_ns": csv_path.stat().st_mtime_ns,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "tsfel_config_sha256": hashlib.sha256(config_payload).hexdigest(),
    }


def cache_is_valid(cached: dict, expected_signature: dict) -> bool:
    return (
        isinstance(cached, dict)
        and cached.get("cache_signature") == expected_signature
        and "windows" in cached
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the TSFEL + HistGradientBoosting LOSO baseline."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/tsfel_histgb"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/tsfel_histgb/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tsfel_histgb"))
    parser.add_argument("--participant-id", type=int, default=4)
    parser.add_argument("--force-reextract", action="store_true")
    return parser.parse_args()


def load_or_extract_subject(
    csv_path: Path,
    subject_id: int,
    cache_dir: Path,
    config: dict,
    force: bool,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"subject_{subject_id}_train_windows.joblib"
    cache_signature = build_cache_signature(csv_path, config)
    if cache_path.exists() and not force:
        cached = joblib.load(cache_path)
        if cache_is_valid(cached, cache_signature):
            windows = cached["windows"]
            print(
                f"Subject {subject_id}: loaded {len(windows.y)} cached windows "+
                f"with {windows.x.shape[1]} features",
                flush=True,
            )
            return windows

    print(f"Subject {subject_id}: reading {csv_path}", flush=True)
    frame = pd.read_csv(csv_path)
    print(f"Subject {subject_id}: extracting TSFEL from {len(frame):,} frames", flush=True)
    windows = extract_labeled_windows(frame, subject_id=subject_id, config=config)
    joblib.dump(
        {"cache_signature": cache_signature, "windows": windows},
        cache_path,
        compress=3,
    )
    print(
        f"Subject {subject_id}: cached {len(windows.y)} windows with {windows.x.shape[1]} features",
        flush=True,
    )
    return windows


def main() -> None:
    args = parse_args()
    subjects = list(dict.fromkeys(args.subjects))
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least two subjects")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = make_tsfel_config()
    subject_windows = {}
    source_files = {}
    for subject_id in subjects:
        source = args.data_dir / f"keypoints_with_labels_{subject_id}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Training CSV not found: {source}")
        source_files[subject_id] = str(source.resolve())
        subject_windows[subject_id] = load_or_extract_subject(
            source,
            subject_id,
            args.cache_dir,
            config,
            args.force_reextract,
        )

    print("Running LOSO evaluation...", flush=True)
    loso = run_loso(subject_windows)
    loso.fold_metrics.to_csv(args.artifact_dir / "loso_fold_metrics.csv", index=False)
    loso.confusion.to_csv(args.artifact_dir / "loso_confusion_matrix.csv")
    for subject_id, matrix in loso.fold_confusions.items():
        matrix.to_csv(args.artifact_dir / f"loso_confusion_matrix_subject_{subject_id}.csv")
    loso.classification_report.to_csv(args.artifact_dir / "loso_classification_report.csv")
    (args.artifact_dir / "loso_summary.json").write_text(
        json.dumps(loso.summary, indent=2), encoding="utf-8"
    )
    print(loso.fold_metrics.to_string(index=False), flush=True)
    for subject_id, matrix in loso.fold_confusions.items():
        print(f"\nConfusion matrix - held-out subject {subject_id}:", flush=True)
        print(matrix.to_string(), flush=True)
    print(json.dumps(loso.summary, indent=2), flush=True)

    print("Fitting final model on all training subjects...", flush=True)
    all_windows = concatenate_window_features([subject_windows[s] for s in subjects])
    final_model = fit_estimator(all_windows)
    artifact = build_artifact(
        final_model,
        all_windows.x.columns,
        metadata={
            "training_subjects": subjects,
            "training_files": source_files,
            "n_training_windows": len(all_windows.y),
            "class_window_counts": all_windows.y.value_counts().sort_index().to_dict(),
            "loso_summary": loso.summary,
        },
    )
    model_name = (
        "keypoint_tsfel_histgb_v7_compatible_"
        + "_".join(map(str, subjects))
        + ".joblib"
    )
    model_path = args.artifact_dir / model_name
    save_artifact(model_path, artifact)
    artifact = load_artifact(model_path)
    print(f"Saved and reloaded model: {model_path.resolve()}", flush=True)

    if args.test_file:
        if not args.test_file.is_file():
            raise FileNotFoundError(f"Test CSV not found: {args.test_file}")
        print(f"Predicting shared test CSV: {args.test_file}", flush=True)
        test_frame = pd.read_csv(args.test_file)
        prediction = predict_frame_labels(test_frame, artifact, args.participant_id)
        paths = write_prediction_outputs(
            args.output_dir, test_frame, prediction, args.participant_id
        )
        distribution = pd.Series(prediction.frame_labels).value_counts().sort_index()
        print("Predicted frame distribution:", flush=True)
        print(distribution.to_string(), flush=True)
        print(f"Filled CSV: {paths.filled.resolve()}", flush=True)
        print(f"Submission CSV: {paths.submission.resolve()}", flush=True)


if __name__ == "__main__":
    main()
