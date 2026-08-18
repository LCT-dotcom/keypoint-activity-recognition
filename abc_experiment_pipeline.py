from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tsfel_histgb_pipeline import (
    CLASSES,
    COORD_COLUMNS,
    FS,
    MAJORITY_THRESHOLD,
    TEST_STRIDE,
    TRAIN_STRIDE,
    WINDOW_SIZE,
    FramePrediction,
    WindowFeatures,
    _extract_tsfel_features,
    _window_metadata,
    add_pose_signals,
    majority_label,
    make_tsfel_config,
    pose_normalize,
    prepare_pose_frame,
    window_starts,
)


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    feature_schema_version: str
    signal_mode: str
    robust_preprocessing: bool
    include_spectral: bool
    random_state: int = 42


EXPERIMENTS = {
    "A": ExperimentDefinition(
        name="A",
        feature_schema_version="abc-a-raw-stat-temp-v1",
        signal_mode="raw",
        robust_preprocessing=False,
        include_spectral=False,
    ),
    "B": ExperimentDefinition(
        name="B",
        feature_schema_version="abc-b-raw-v7-stat-temp-v1",
        signal_mode="raw_v7",
        robust_preprocessing=False,
        include_spectral=False,
    ),
    "C": ExperimentDefinition(
        name="C",
        feature_schema_version="abc-c-robust-symmetric-spectral-v1",
        signal_mode="robust_symmetric",
        robust_preprocessing=True,
        include_spectral=True,
    ),
}


B_HANDCRAFTED_SIGNALS = [
    "dist_shoulders",
    "dist_hips",
    "dist_knees",
    "dist_ankles",
    "dist_wrist_to_hip",
    "dist_lw_nose",
    "dist_rw_nose",
    "dist_lw_ear",
    "dist_rw_ear",
    "dist_la_lhip",
    "dist_ra_rhip",
    "dist_lw_floor_norm",
    "dist_rw_floor_norm",
    "angle_elbow_l",
    "angle_elbow_r",
    "angle_knee_l",
    "angle_knee_r",
    "angle_hip_l",
    "angle_hip_r",
    "angle_shoulder_tilt",
    "angle_torso",
    "pelvis_y",
    "wrist_y_diff",
    "vel_dist_lw_nose",
    "vel_dist_rw_nose",
    "vel_head_nod",
    "vel_left_wrist",
    "vel_right_wrist",
    "left_wrist_jerk",
    "right_wrist_jerk",
    "jerk_nose_y",
    "vel_com",
    "total_movement",
    "min_dist_wrist_head",
    "static_wrist",
    "elbow_flex",
]


C_TARGETED_SIGNALS = [
    "dist_wrists",
    "dist_lw_lhip",
    "dist_rw_rhip",
    "dist_lw_lknee",
    "dist_rw_rknee",
    "max_dist_wrist_head",
    "wrist_head_distance_diff",
    "both_hands_near_head",
    "one_hand_near_head",
    "wrist_speed_diff",
    "wrist_speed_max",
    "wrist_motion_sync",
    "mean_knee_angle",
    "knee_angle_asymmetry",
    "torso_height",
]


def get_experiment(name: str) -> ExperimentDefinition:
    key = str(name).upper()
    if key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    return EXPERIMENTS[key]


def _robust_pose_normalize(df: pd.DataFrame) -> pd.DataFrame:
    smoothed = df.copy()
    smoothed[COORD_COLUMNS] = (
        smoothed[COORD_COLUMNS]
        .rolling(window=5, center=True, min_periods=1)
        .median()
    )

    hip_mid_x = (smoothed["left_hip_x"] + smoothed["right_hip_x"]) / 2
    hip_mid_y = (smoothed["left_hip_y"] + smoothed["right_hip_y"]) / 2
    shoulder_mid_x = (smoothed["left_shoulder_x"] + smoothed["right_shoulder_x"]) / 2
    shoulder_mid_y = (smoothed["left_shoulder_y"] + smoothed["right_shoulder_y"]) / 2
    shoulder_width = np.sqrt(
        (smoothed["left_shoulder_x"] - smoothed["right_shoulder_x"]) ** 2
        + (smoothed["left_shoulder_y"] - smoothed["right_shoulder_y"]) ** 2
    )
    torso_length = np.sqrt(
        (shoulder_mid_x - hip_mid_x) ** 2 + (shoulder_mid_y - hip_mid_y) ** 2
    )
    scale = shoulder_width.where(shoulder_width > 1e-6, torso_length)
    scale = scale.rolling(window=15, center=True, min_periods=1).median()
    finite = scale.replace([np.inf, -np.inf], np.nan).dropna()
    fallback = float(finite.median()) if not finite.empty else 1.0
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(fallback).clip(lower=1e-6)

    for column in (c for c in COORD_COLUMNS if c.endswith("_x")):
        smoothed[column] = (smoothed[column] - hip_mid_x) / scale
    for column in (c for c in COORD_COLUMNS if c.endswith("_y")):
        smoothed[column] = (smoothed[column] - hip_mid_y) / scale
    return smoothed


def normalize_for_experiment(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    definition = get_experiment(experiment)
    if definition.robust_preprocessing:
        return _robust_pose_normalize(df)
    return pose_normalize(df)


def _distance(df: pd.DataFrame, first: str, second: str) -> pd.Series:
    return np.sqrt(
        (df[f"{first}_x"] - df[f"{second}_x"]) ** 2
        + (df[f"{first}_y"] - df[f"{second}_y"]) ** 2
    )


def build_experiment_signals(
    normalized: pd.DataFrame,
    experiment: str,
) -> tuple[pd.DataFrame, list[str]]:
    definition = get_experiment(experiment)
    if definition.signal_mode == "raw":
        out = normalized.copy()
        out[COORD_COLUMNS] = out[COORD_COLUMNS].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        return out, COORD_COLUMNS.copy()

    out, _ = add_pose_signals(normalized)
    signal_columns = COORD_COLUMNS.copy() + B_HANDCRAFTED_SIGNALS.copy()
    if definition.signal_mode == "robust_symmetric":
        out["dist_wrists"] = _distance(out, "left_wrist", "right_wrist")
        out["dist_lw_lhip"] = _distance(out, "left_wrist", "left_hip")
        out["dist_rw_rhip"] = _distance(out, "right_wrist", "right_hip")
        out["dist_lw_lknee"] = _distance(out, "left_wrist", "left_knee")
        out["dist_rw_rknee"] = _distance(out, "right_wrist", "right_knee")
        out["max_dist_wrist_head"] = out[["dist_lw_nose", "dist_rw_nose"]].max(axis=1)
        out["wrist_head_distance_diff"] = (
            out["dist_lw_nose"] - out["dist_rw_nose"]
        ).abs()
        left_near = out["dist_lw_nose"].lt(1.0)
        right_near = out["dist_rw_nose"].lt(1.0)
        out["both_hands_near_head"] = (left_near & right_near).astype(float)
        out["one_hand_near_head"] = (left_near ^ right_near).astype(float)
        out["wrist_speed_diff"] = (
            out["vel_left_wrist"] - out["vel_right_wrist"]
        ).abs()
        out["wrist_speed_max"] = out[["vel_left_wrist", "vel_right_wrist"]].max(axis=1)
        out["wrist_motion_sync"] = out["vel_left_wrist"] * out["vel_right_wrist"]
        out["mean_knee_angle"] = (out["angle_knee_l"] + out["angle_knee_r"]) / 2
        out["knee_angle_asymmetry"] = (
            out["angle_knee_l"] - out["angle_knee_r"]
        ).abs()
        shoulder_mid_y = (out["left_shoulder_y"] + out["right_shoulder_y"]) / 2
        hip_mid_y = (out["left_hip_y"] + out["right_hip_y"]) / 2
        out["torso_height"] = (shoulder_mid_y - hip_mid_y).abs()
        signal_columns += C_TARGETED_SIGNALS

    out[signal_columns] = (
        out[signal_columns]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0)
    )
    return out, signal_columns


def make_experiment_tsfel_config(experiment: str) -> dict:
    definition = get_experiment(experiment)
    config = make_tsfel_config()
    if not definition.include_spectral:
        config.pop("spectral")
    return config


def derive_experiment_a_windows(windows_b: WindowFeatures) -> WindowFeatures:
    prefixes = tuple(f"{column}_" for column in COORD_COLUMNS)
    columns = [column for column in windows_b.x.columns if column.startswith(prefixes)]
    if not columns:
        raise ValueError("Experiment B windows contain no raw-coordinate TSFEL features")
    return WindowFeatures(
        x=windows_b.x[columns].copy(),
        y=windows_b.y.copy(),
        meta=windows_b.meta.copy(),
    )


def extract_experiment_labeled_windows(
    df: pd.DataFrame,
    subject_id: int,
    experiment: str,
    window_size: int = WINDOW_SIZE,
    stride: int = TRAIN_STRIDE,
    majority_threshold: float = MAJORITY_THRESHOLD,
    fs: int = FS,
) -> WindowFeatures:
    prepared = prepare_pose_frame(df)
    if "Action Label" not in prepared.columns:
        raise ValueError("Training CSV must contain an 'Action Label' column")
    normalized = normalize_for_experiment(prepared, experiment)
    enriched, signal_columns = build_experiment_signals(normalized, experiment)
    valid_starts: list[int] = []
    labels: list[str] = []
    for start in window_starts(len(prepared), window_size, stride, cover_tail=False):
        label = majority_label(
            prepared["Action Label"].iloc[start : start + window_size],
            threshold=majority_threshold,
        )
        if label is not None:
            valid_starts.append(start)
            labels.append(label)
    if not valid_starts:
        raise ValueError(f"Subject {subject_id} produced no valid labeled windows")
    x = _extract_tsfel_features(
        enriched[signal_columns],
        valid_starts,
        window_size,
        make_experiment_tsfel_config(experiment),
        fs,
    )
    return WindowFeatures(
        x=x,
        y=pd.Series(labels, name="Action Label", dtype="string"),
        meta=_window_metadata(prepared, subject_id, valid_starts, window_size),
    )


def extract_experiment_unlabeled_windows(
    df: pd.DataFrame,
    subject_id: int,
    experiment: str,
    window_size: int = WINDOW_SIZE,
    stride: int = TEST_STRIDE,
    fs: int = FS,
) -> WindowFeatures:
    prepared = prepare_pose_frame(df)
    normalized = normalize_for_experiment(prepared, experiment)
    enriched, signal_columns = build_experiment_signals(normalized, experiment)
    starts = window_starts(len(prepared), window_size, stride, cover_tail=True)
    x = _extract_tsfel_features(
        enriched[signal_columns],
        starts,
        window_size,
        make_experiment_tsfel_config(experiment),
        fs,
    )
    return WindowFeatures(
        x=x,
        y=pd.Series([None] * len(starts), name="Action Label", dtype="object"),
        meta=_window_metadata(prepared, subject_id, starts, window_size),
    )


def build_experiment_cache_signature(csv_path: Path, experiment: str) -> dict:
    definition = get_experiment(experiment)
    config_payload = json.dumps(
        make_experiment_tsfel_config(experiment), sort_keys=True, default=str
    ).encode("utf-8")
    return {
        "path": str(csv_path.resolve()),
        "size": csv_path.stat().st_size,
        "mtime_ns": csv_path.stat().st_mtime_ns,
        "experiment": definition.name,
        "feature_schema_version": definition.feature_schema_version,
        "tsfel_config_sha256": hashlib.sha256(config_payload).hexdigest(),
    }


def experiment_cache_path(cache_dir: Path, subject_id: int, experiment: str) -> Path:
    return cache_dir / f"abc_{experiment.lower()}_subject_{subject_id}_windows.joblib"


def save_experiment_cache(
    cache_dir: Path,
    csv_path: Path,
    subject_id: int,
    experiment: str,
    windows: WindowFeatures,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = experiment_cache_path(cache_dir, subject_id, experiment)
    joblib.dump(
        {
            "cache_signature": build_experiment_cache_signature(csv_path, experiment),
            "windows": windows,
        },
        path,
        compress=3,
    )
    return path


def load_or_extract_experiment_subject(
    csv_path: Path,
    subject_id: int,
    experiment: str,
    cache_dir: Path,
    force: bool = False,
) -> WindowFeatures:
    path = experiment_cache_path(cache_dir, subject_id, experiment)
    signature = build_experiment_cache_signature(csv_path, experiment)
    if path.exists() and not force:
        cached = joblib.load(path)
        if cached.get("cache_signature") == signature and "windows" in cached:
            return cached["windows"]
    frame = pd.read_csv(csv_path)
    windows = extract_experiment_labeled_windows(frame, subject_id, experiment)
    save_experiment_cache(cache_dir, csv_path, subject_id, experiment, windows)
    return windows


def build_experiment_artifact(
    model,
    feature_columns: list[str] | pd.Index,
    experiment: str,
    metadata: dict,
) -> dict:
    definition = get_experiment(experiment)
    return {
        "format_version": 1,
        "architecture": "TSFEL+HistGradientBoosting",
        "experiment": definition.name,
        "experiment_definition": asdict(definition),
        "feature_schema_version": definition.feature_schema_version,
        "model": model,
        "feature_columns": list(feature_columns),
        "classes": CLASSES.copy(),
        "tsfel_config": make_experiment_tsfel_config(experiment),
        "preprocessing": {
            "fps": FS,
            "window_size": WINDOW_SIZE,
            "train_stride": TRAIN_STRIDE,
            "test_stride": TEST_STRIDE,
            "majority_threshold": MAJORITY_THRESHOLD,
            "experiment": definition.name,
        },
        "metadata": dict(metadata),
    }


def save_experiment_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_experiment_artifact(path: Path) -> dict:
    artifact = joblib.load(path)
    required = {
        "model",
        "feature_columns",
        "classes",
        "experiment",
        "feature_schema_version",
        "preprocessing",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Incompatible experiment artifact; missing keys: {missing}")
    definition = get_experiment(artifact["experiment"])
    if artifact["feature_schema_version"] != definition.feature_schema_version:
        raise ValueError("Experiment artifact feature schema is incompatible")
    if artifact["classes"] != CLASSES:
        raise ValueError("Experiment artifact class order is incompatible")
    return artifact


def predict_experiment_frame_labels(
    df: pd.DataFrame,
    artifact: dict,
    participant_id: int,
) -> FramePrediction:
    definition = get_experiment(artifact["experiment"])
    if artifact["feature_schema_version"] != definition.feature_schema_version:
        raise ValueError("Experiment artifact feature schema is incompatible")
    windows = extract_experiment_unlabeled_windows(
        df,
        participant_id,
        definition.name,
        window_size=int(artifact["preprocessing"]["window_size"]),
        stride=int(artifact["preprocessing"]["test_stride"]),
        fs=int(artifact["preprocessing"]["fps"]),
    )
    expected = list(artifact["feature_columns"])
    missing = sorted(set(expected) - set(windows.x.columns))
    if missing:
        raise ValueError(f"Experiment test features are missing {len(missing)} columns")
    x = windows.x.reindex(columns=expected)
    model = artifact["model"]
    raw_probabilities = model.predict_proba(x)
    probabilities = np.zeros((len(x), len(CLASSES)), dtype=float)
    for index, label in enumerate(model.classes_):
        probabilities[:, CLASSES.index(str(label))] = raw_probabilities[:, index]

    frame_scores = np.zeros((len(df), len(CLASSES)), dtype=float)
    frame_votes = np.zeros(len(df), dtype=int)
    for window_index, row in windows.meta.reset_index(drop=True).iterrows():
        start = int(row["start_pos"])
        end = int(row["end_pos"])
        frame_scores[start:end] += probabilities[window_index]
        frame_votes[start:end] += 1
    if np.any(frame_votes == 0):
        raise RuntimeError("Experiment windowing left rows without predictions")
    frame_probabilities = frame_scores / frame_votes[:, None]
    label_indices = frame_probabilities.argmax(axis=1)
    labels = np.asarray(CLASSES, dtype=object)
    return FramePrediction(
        frame_labels=labels[label_indices],
        confidence=frame_probabilities[np.arange(len(df)), label_indices],
        window_predictions=labels[probabilities.argmax(axis=1)],
        window_probabilities=probabilities,
        meta=windows.meta.copy(),
    )


def select_winning_experiment(comparison: pd.DataFrame) -> str:
    required = {
        "experiment",
        "pooled_abnormal_f1",
        "worst_subject_accuracy",
        "pooled_macro_f1",
    }
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"Comparison table is missing columns: {missing}")
    ordered = comparison.sort_values(
        ["pooled_abnormal_f1", "worst_subject_accuracy", "pooled_macro_f1", "experiment"],
        ascending=[False, False, False, True],
    )
    return str(ordered.iloc[0]["experiment"])
