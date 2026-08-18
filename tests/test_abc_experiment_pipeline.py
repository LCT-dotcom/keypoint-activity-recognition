from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abc_experiment_pipeline import (
    EXPERIMENTS,
    build_experiment_cache_signature,
    build_experiment_signals,
    derive_experiment_a_windows,
    make_experiment_tsfel_config,
    normalize_for_experiment,
    select_winning_experiment,
)
from tsfel_histgb_pipeline import COORD_COLUMNS, JOINTS, WindowFeatures, prepare_pose_frame


def make_pose(rows: int = 300) -> pd.DataFrame:
    t = np.arange(rows, dtype=float)
    data = {"frame_id": np.arange(rows)}
    for index, joint in enumerate(JOINTS):
        data[f"{joint}_x"] = 100 + index * 4 + 0.08 * t + np.sin(t / 12)
        data[f"{joint}_y"] = 200 + index * 3 + np.cos(t / 10)
    data["Action Label"] = "Walking"
    return pd.DataFrame(data)


def test_a_and_b_share_preprocessing_but_b_adds_handcrafted_signals() -> None:
    prepared = prepare_pose_frame(make_pose())

    normalized_a = normalize_for_experiment(prepared, "A")
    normalized_b = normalize_for_experiment(prepared, "B")
    _, columns_a = build_experiment_signals(normalized_a, "A")
    enriched_b, columns_b = build_experiment_signals(normalized_b, "B")

    assert normalized_a[COORD_COLUMNS].equals(normalized_b[COORD_COLUMNS])
    assert columns_a == COORD_COLUMNS
    assert set(columns_a) < set(columns_b)
    assert {
        "dist_lw_nose",
        "angle_elbow_l",
        "vel_head_nod",
        "total_movement",
    } <= set(columns_b)
    assert np.isfinite(enriched_b[columns_b].to_numpy()).all()


def test_c_adds_targeted_symmetric_features_and_robust_preprocessing() -> None:
    prepared = prepare_pose_frame(make_pose())

    normalized_b = normalize_for_experiment(prepared, "B")
    normalized_c = normalize_for_experiment(prepared, "C")
    enriched_c, columns_c = build_experiment_signals(normalized_c, "C")

    assert not normalized_b[COORD_COLUMNS].equals(normalized_c[COORD_COLUMNS])
    assert {
        "dist_wrists",
        "both_hands_near_head",
        "one_hand_near_head",
        "wrist_head_distance_diff",
        "wrist_speed_diff",
        "wrist_motion_sync",
        "mean_knee_angle",
    } <= set(columns_c)
    assert np.isfinite(enriched_c[columns_c].to_numpy()).all()


def test_tsfel_domains_are_controlled_between_experiments() -> None:
    config_a = make_experiment_tsfel_config("A")
    config_b = make_experiment_tsfel_config("B")
    config_c = make_experiment_tsfel_config("C")

    assert config_a == config_b
    assert set(config_a) == {"statistical", "temporal"}
    assert set(config_c) == {"statistical", "temporal", "spectral"}
    assert EXPERIMENTS["A"].random_state == EXPERIMENTS["B"].random_state == EXPERIMENTS["C"].random_state == 42


def test_winner_selection_prioritizes_abnormal_f1_then_hard_subject() -> None:
    comparison = pd.DataFrame(
        [
            {"experiment": "A", "pooled_abnormal_f1": 0.75, "worst_subject_accuracy": 0.45, "pooled_macro_f1": 0.70},
            {"experiment": "B", "pooled_abnormal_f1": 0.80, "worst_subject_accuracy": 0.40, "pooled_macro_f1": 0.72},
            {"experiment": "C", "pooled_abnormal_f1": 0.80, "worst_subject_accuracy": 0.50, "pooled_macro_f1": 0.69},
        ]
    )

    assert select_winning_experiment(comparison) == "C"


def test_a_windows_can_be_derived_from_b_without_reextracting_tsfel() -> None:
    windows_b = WindowFeatures(
        x=pd.DataFrame(
            {
                "nose_x_Mean": [1.0, 2.0],
                "right_ankle_y_Standard deviation": [0.1, 0.2],
                "dist_lw_nose_Mean": [0.5, 0.6],
            }
        ),
        y=pd.Series(["Walking", "Biting"]),
        meta=pd.DataFrame({"subject_id": [1, 1]}),
    )

    windows_a = derive_experiment_a_windows(windows_b)

    assert windows_a.x.columns.tolist() == [
        "nose_x_Mean",
        "right_ankle_y_Standard deviation",
    ]
    assert windows_a.y.equals(windows_b.y)


def test_cache_signatures_are_isolated_by_experiment() -> None:
    source = Path("artifacts/test_tmp/abc_subject.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("frame_id\n0\n", encoding="utf-8")

    signature_b = build_experiment_cache_signature(source, "B")
    signature_c = build_experiment_cache_signature(source, "C")

    assert signature_b["experiment"] == "B"
    assert signature_c["experiment"] == "C"
    assert signature_b["feature_schema_version"] != signature_c["feature_schema_version"]
