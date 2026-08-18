from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from train_tsfel_histgb import build_cache_signature, cache_is_valid

from tsfel_histgb_pipeline import (
    FEATURE_SCHEMA_VERSION,
    COORD_COLUMNS,
    JOINTS,
    TRAIN_STRIDE,
    FramePrediction,
    WindowFeatures,
    add_pose_signals,
    clean_labels,
    build_artifact,
    combine_loso_folds,
    evaluate_predictions,
    extract_labeled_windows,
    extract_unlabeled_windows,
    fit_estimator,
    load_artifact,
    make_tsfel_config,
    majority_label,
    predict_frame_labels,
    pose_normalize,
    prepare_pose_frame,
    run_loso,
    run_loso_fold,
    save_artifact,
    validate_pose_columns,
    write_prediction_outputs,
)


def make_synthetic_pose(rows: int = 300) -> pd.DataFrame:
    t = np.arange(rows, dtype=float)
    data: dict[str, np.ndarray | str] = {"frame_id": np.arange(rows)}
    for joint_index, joint in enumerate(JOINTS):
        data[f"{joint}_x"] = 100 + joint_index * 3 + 0.05 * t
        data[f"{joint}_y"] = 200 + joint_index * 2 + np.sin(t / 15)
    data["Action Label"] = "Walking"
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_pose() -> pd.DataFrame:
    return make_synthetic_pose(300)


def test_clean_labels_merges_throwing_and_excludes_none() -> None:
    values = pd.Series(["Throwing", "Throwing things", None, "None", "Walking"])

    assert clean_labels(values).tolist() == [
        "Throwing things",
        "Throwing things",
        "None",
        "None",
        "Walking",
    ]


def test_pose_normalize_centers_hip_midpoint(synthetic_pose: pd.DataFrame) -> None:
    normalized = pose_normalize(prepare_pose_frame(synthetic_pose))

    assert np.allclose((normalized.left_hip_x + normalized.right_hip_x) / 2, 0)
    assert np.allclose((normalized.left_hip_y + normalized.right_hip_y) / 2, 0)


def test_validate_pose_columns_names_missing_column(synthetic_pose: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="right_ankle_y"):
        validate_pose_columns(synthetic_pose.drop(columns=["right_ankle_y"]))


def test_add_pose_signals_returns_finite_stable_columns(synthetic_pose: pd.DataFrame) -> None:
    normalized = pose_normalize(prepare_pose_frame(synthetic_pose))

    enriched, signal_columns = add_pose_signals(normalized)

    assert FEATURE_SCHEMA_VERSION.startswith("v7-compatible")
    assert signal_columns[:4] == [
        "dist_shoulders",
        "dist_hips",
        "dist_knees",
        "dist_ankles",
    ]
    assert {
        "dist_lw_nose",
        "dist_lw_floor_norm",
        "angle_elbow_l",
        "angle_shoulder_tilt",
        "vel_head_nod",
        "right_wrist_jerk",
        "jerk_nose_y",
        "hand_near_head",
        "micro_bite",
        "strong_bite",
        "knee_extend",
        "static_wrist",
        "ratio_rwjerk_comjerk",
        "total_movement",
    } <= set(signal_columns)
    assert not set(COORD_COLUMNS) & set(signal_columns)
    assert np.isfinite(enriched[signal_columns].to_numpy()).all()


def test_majority_label_rejects_transition_window() -> None:
    assert majority_label(["Walking"] * 105 + ["None"] * 45) == "Walking"
    assert majority_label(["Walking"] * 104 + ["None"] * 46) is None


def test_tsfel_config_uses_only_builtin_domains() -> None:
    config = make_tsfel_config()

    assert "Custom" not in config
    assert set(config) == {"statistical", "temporal", "spectral"}
    enabled = {
        name
        for features in config.values()
        for name, specification in features.items()
        if specification.get("use") == "yes"
    }
    assert {"Mean", "Standard deviation", "Area under the curve", "Mean absolute diff"} <= enabled
    assert {
        "Fundamental frequency",
        "Spectral centroid",
        "Spectral entropy",
    } <= enabled


def test_cache_signature_rejects_previous_feature_schema() -> None:
    source = Path("artifacts/test_tmp/cache_source.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("frame_id\n0\n", encoding="utf-8")
    expected = build_cache_signature(source, make_tsfel_config())

    stale = {
        "cache_signature": {**expected, "feature_schema_version": "reduced-v8"},
        "windows": object(),
    }
    current = {"cache_signature": expected, "windows": object()}

    assert expected["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert not cache_is_valid(stale, expected)
    assert cache_is_valid(current, expected)


def test_labeled_windows_apply_majority_threshold() -> None:
    frame = make_synthetic_pose(300)
    frame["Action Label"] = ["Walking"] * 149 + ["None"] + ["Walking"] * 90 + ["Biting"] * 60

    windows = extract_labeled_windows(
        frame,
        subject_id=1,
        config=make_tsfel_config(),
        window_size=150,
        stride=150,
    )

    assert windows.y.tolist() == ["Walking"]
    assert windows.meta[["start_pos", "end_pos"]].to_dict("records") == [{"start_pos": 0, "end_pos": 150}]


def test_unlabeled_windows_cover_tail() -> None:
    frame = make_synthetic_pose(301).drop(columns=["Action Label"])

    windows = extract_unlabeled_windows(
        frame,
        subject_id=4,
        config=make_tsfel_config(),
        window_size=150,
        stride=150,
    )

    assert windows.meta.iloc[-1].end_pos == 301
    assert windows.meta.iloc[-1].start_pos == 151
    assert len(windows.x) == 3


def make_tiny_window_features(subject_id: int, rows: int = 24) -> WindowFeatures:
    labels = np.asarray(["Walking", "Biting"] * (rows // 2), dtype=object)
    base = np.arange(rows, dtype=float)
    x = pd.DataFrame(
        {
            "signal__mean": (labels == "Biting").astype(float) + subject_id * 0.01,
            "signal__std": base / rows + subject_id * 0.02,
        }
    )
    return WindowFeatures(
        x=x,
        y=pd.Series(labels, name="Action Label", dtype="string"),
        meta=pd.DataFrame(
            {
                "subject_id": [subject_id] * rows,
                "start_pos": np.arange(rows) * TRAIN_STRIDE,
            }
        ),
    )


def test_joblib_round_trip_reproduces_predictions() -> None:
    windows = make_tiny_window_features(1)
    model = fit_estimator(windows, random_state=7)
    before = model.predict(windows.x)
    path = Path("artifacts/test_tmp/model.joblib")

    save_artifact(path, build_artifact(model, windows.x.columns, metadata={"test": True}))
    loaded = load_artifact(path)
    after = loaded["model"].predict(windows.x)

    assert np.array_equal(before, after)
    assert loaded["classes"] == [
        "Attacking",
        "Biting",
        "Eating snacks",
        "Head banging",
        "Sitting quietly",
        "Throwing things",
        "Using phone",
        "Walking",
    ]


def test_abnormal_metrics_use_four_defined_classes() -> None:
    metrics = evaluate_predictions(
        ["Walking", "Attacking", "Biting"],
        ["Walking", "Attacking", "Walking"],
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["abnormal_recall"] == pytest.approx(0.5)


def test_run_loso_holds_out_each_subject_once() -> None:
    subject_windows = {subject: make_tiny_window_features(subject) for subject in (1, 2, 3, 5)}

    result = run_loso(subject_windows, random_state=11)

    assert result.fold_metrics.held_out_subject.tolist() == [1, 2, 3, 5]
    assert result.fold_metrics.n_train_windows.tolist() == [72, 72, 72, 72]
    assert result.fold_metrics.n_test_windows.tolist() == [12, 12, 12, 12]
    assert len(result.y_true) == 48
    assert len(result.y_pred) == 48
    assert set(result.fold_confusions) == {1, 2, 3, 5}
    for matrix in result.fold_confusions.values():
        assert matrix.shape == (8, 8)
        assert matrix.to_numpy().sum() == 12


def test_individual_loso_folds_can_be_run_and_combined() -> None:
    subject_windows = {subject: make_tiny_window_features(subject) for subject in (1, 2, 3, 5)}

    folds = [run_loso_fold(subject_windows, subject, random_state=11) for subject in (1, 2, 3, 5)]
    result = combine_loso_folds(folds)

    assert [fold.held_out_subject for fold in folds] == [1, 2, 3, 5]
    assert all(fold.n_train_windows == 72 for fold in folds)
    assert all(fold.n_test_windows == 12 for fold in folds)
    assert result.fold_metrics.held_out_subject.tolist() == [1, 2, 3, 5]


@pytest.fixture
def fake_artifact() -> dict:
    frame = make_synthetic_pose(300)
    frame.loc[:149, "Action Label"] = "Walking"
    frame.loc[150:, "Action Label"] = "Biting"
    windows = extract_labeled_windows(
        frame,
        subject_id=1,
        config=make_tsfel_config(),
        window_size=150,
        stride=150,
    )
    model = fit_estimator(windows)
    return build_artifact(model, windows.x.columns, metadata={"test": True})


@pytest.fixture
def fake_prediction() -> FramePrediction:
    return FramePrediction(
        frame_labels=np.asarray(["Walking"] * 301, dtype=object),
        confidence=np.ones(301),
        window_predictions=np.asarray(["Walking", "Walking", "Walking"], dtype=object),
        window_probabilities=np.ones((3, 8)) / 8,
        meta=pd.DataFrame({"start_pos": [0, 150, 151], "end_pos": [150, 300, 301]}),
    )


def test_frame_predictions_cover_every_input_row(fake_artifact: dict) -> None:
    frame = make_synthetic_pose(301).drop(columns=["Action Label"])

    prediction = predict_frame_labels(frame, fake_artifact, participant_id=4)

    assert len(prediction.frame_labels) == 301
    assert prediction.confidence.shape == (301,)
    assert set(prediction.frame_labels) <= set(fake_artifact["classes"])


def test_submission_has_exact_three_columns(fake_prediction: FramePrediction) -> None:
    frame = make_synthetic_pose(301).drop(columns=["Action Label"])
    output_dir = Path("artifacts/test_tmp/prediction_outputs")

    paths = write_prediction_outputs(output_dir, frame, fake_prediction, participant_id=4)
    filled = pd.read_csv(paths.filled)
    submission = pd.read_csv(paths.submission)

    assert len(filled) == len(submission) == 301
    assert {"predicted_label", "prediction_confidence"} <= set(filled.columns)
    assert submission.columns.tolist() == ["participant_id", "timestamp", "predicted_label"]
    assert submission["timestamp"].tolist() == frame.frame_id.tolist()
