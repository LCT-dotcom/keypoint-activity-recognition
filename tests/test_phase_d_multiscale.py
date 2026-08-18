from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abc_experiment_pipeline import normalize_for_experiment
from phase_d_multiscale import (
    MULTISCALE_FUSION_WEIGHTS,
    build_compact_signals,
    compact_cache_path,
    extract_compact_windows,
    fuse_scale_probabilities,
    label_compact_windows,
    subsample_compact_windows,
)
from tsfel_histgb_pipeline import JOINTS, prepare_pose_frame


def make_pose(rows: int = 360) -> pd.DataFrame:
    time = np.arange(rows, dtype=float)
    data: dict[str, np.ndarray] = {"frame_id": np.arange(rows)}
    for index, joint in enumerate(JOINTS):
        data[f"{joint}_x"] = 100 + index * 3 + 0.02 * time + np.sin(time / 15 + index)
        data[f"{joint}_y"] = 200 + index * 2 + np.cos(time / 13 + index)
    data["Action Label"] = np.where(time < rows / 2, "Walking", "Attacking")
    return pd.DataFrame(data)


def test_compact_signals_include_bones_angles_velocity_and_acceleration() -> None:
    normalized = normalize_for_experiment(prepare_pose_frame(make_pose()), "C")

    signals = build_compact_signals(normalized)

    assert {
        "left_upper_arm_dx",
        "left_upper_arm_length",
        "left_elbow_angle",
        "left_wrist_speed",
        "left_wrist_acceleration",
        "left_wrist_to_nose",
    } <= set(signals.columns)
    assert np.isfinite(signals.to_numpy()).all()


def test_compact_window_extraction_has_stable_summary_schema_and_metadata() -> None:
    windows = extract_compact_windows(
        make_pose(),
        subject_id=7,
        window_size=60,
        stride=30,
        majority_threshold=0.70,
        labeled=True,
    )

    assert not windows.x.empty
    assert windows.x.columns.is_unique
    assert {"left_wrist_speed__mean", "left_wrist_speed__slope", "left_elbow_angle__iqr"} <= set(windows.x)
    assert windows.meta["subject_id"].eq(7).all()
    assert windows.meta.iloc[0]["start_pos"] == 0
    assert windows.meta.iloc[0]["end_pos"] == 60
    assert np.isfinite(windows.x.to_numpy()).all()


def test_unlabeled_inference_windows_do_not_depend_on_ground_truth_labels() -> None:
    first = make_pose()
    second = first.copy()
    second["Action Label"] = np.where(
        np.arange(len(second)) % 2 == 0, "Biting nails", "Throwing things"
    )

    first_windows = extract_compact_windows(
        first, subject_id=7, window_size=60, stride=15, labeled=False
    )
    second_windows = extract_compact_windows(
        second, subject_id=7, window_size=60, stride=15, labeled=False
    )

    pd.testing.assert_frame_equal(first_windows.x, second_windows.x)
    pd.testing.assert_frame_equal(first_windows.meta, second_windows.meta)
    assert first_windows.y.isna().all() and second_windows.y.isna().all()


def test_training_labels_are_attached_after_feature_extraction() -> None:
    pose = make_pose()
    unlabeled = extract_compact_windows(
        pose, subject_id=7, window_size=60, stride=15, labeled=False
    )
    stride_30 = subsample_compact_windows(
        unlabeled, stride=30, n_frames=len(pose), cover_tail=False
    )

    labeled = label_compact_windows(
        stride_30, pose["Action Label"].to_numpy(dtype=object), majority_threshold=0.85
    )

    assert not labeled.x.empty
    assert labeled.y.notna().all()
    assert set(labeled.y) <= {"Walking", "Attacking"}
    assert (labeled.meta["start_pos"] % 30 == 0).all()


def test_inference_subsampling_keeps_tail_coverage() -> None:
    pose = make_pose(rows=367)
    base = extract_compact_windows(
        pose, subject_id=7, window_size=60, stride=15, labeled=False
    )

    inference = subsample_compact_windows(
        base, stride=30, n_frames=len(pose), cover_tail=True
    )

    assert inference.meta.iloc[-1]["end_pos"] == len(pose)
    assert inference.meta.iloc[-1]["start_pos"] == len(pose) - 60


def test_scale_cache_paths_cannot_collide() -> None:
    cache_root = Path("synthetic-cache")
    path_60 = compact_cache_path(cache_root, 1, window_size=60, stride=30, majority_threshold=0.70)
    path_150 = compact_cache_path(cache_root, 1, window_size=150, stride=75, majority_threshold=0.70)
    path_threshold = compact_cache_path(cache_root, 1, window_size=60, stride=30, majority_threshold=0.85)

    assert len({path_60, path_150, path_threshold}) == 3


def test_fusion_uses_only_declared_weights_and_normalizes_rows() -> None:
    probabilities = {
        60: np.array([[0.8, 0.2], [0.7, 0.3]]),
        150: np.array([[0.4, 0.6], [0.6, 0.4]]),
        300: np.array([[0.1, 0.9], [0.2, 0.8]]),
    }
    weights = MULTISCALE_FUSION_WEIGHTS[3]

    fused = fuse_scale_probabilities(probabilities, weights)

    np.testing.assert_allclose(fused.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="declared fusion grid"):
        fuse_scale_probabilities(probabilities, (0.1, 0.1, 0.8))


def test_fusion_rejects_mismatched_frame_counts() -> None:
    probabilities = {
        60: np.ones((2, 2)) / 2,
        150: np.ones((3, 2)) / 2,
        300: np.ones((2, 2)) / 2,
    }

    with pytest.raises(ValueError, match="same shape"):
        fuse_scale_probabilities(probabilities, MULTISCALE_FUSION_WEIGHTS[3])
