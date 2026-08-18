from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from abc_experiment_pipeline import normalize_for_experiment
from tsfel_histgb_pipeline import (
    COORD_COLUMNS,
    FS,
    JOINTS,
    WindowFeatures,
    _window_metadata,
    majority_label,
    prepare_pose_frame,
    window_starts,
)


COMPACT_SCHEMA_VERSION = "phase-d-compact-v1"
MULTISCALE_WINDOWS = (60, 150, 300)
MULTISCALE_FUSION_WEIGHTS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1 / 3, 1 / 3, 1 / 3),
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.2, 0.6, 0.2),
    (0.5, 0.3, 0.2),
    (0.2, 0.3, 0.5),
)

BONES = (
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
    ("left_torso", "left_shoulder", "left_hip"),
    ("right_torso", "right_shoulder", "right_hip"),
    ("shoulders", "left_shoulder", "right_shoulder"),
    ("hips", "left_hip", "right_hip"),
    ("left_thigh", "left_hip", "left_knee"),
    ("left_shin", "left_knee", "left_ankle"),
    ("right_thigh", "right_hip", "right_knee"),
    ("right_shin", "right_knee", "right_ankle"),
    ("nose_left_wrist", "nose", "left_wrist"),
    ("nose_right_wrist", "nose", "right_wrist"),
)

ANGLES = (
    ("left_elbow_angle", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_elbow_angle", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_knee_angle", "left_hip", "left_knee", "left_ankle"),
    ("right_knee_angle", "right_hip", "right_knee", "right_ankle"),
    ("left_hip_angle", "left_shoulder", "left_hip", "left_knee"),
    ("right_hip_angle", "right_shoulder", "right_hip", "right_knee"),
)


def _distance(frame: pd.DataFrame, first: str, second: str) -> pd.Series:
    return np.hypot(
        frame[f"{first}_x"] - frame[f"{second}_x"],
        frame[f"{first}_y"] - frame[f"{second}_y"],
    )


def _joint_angle(frame: pd.DataFrame, first: str, center: str, third: str) -> pd.Series:
    first_vector = np.column_stack(
        [
            frame[f"{first}_x"] - frame[f"{center}_x"],
            frame[f"{first}_y"] - frame[f"{center}_y"],
        ]
    )
    third_vector = np.column_stack(
        [
            frame[f"{third}_x"] - frame[f"{center}_x"],
            frame[f"{third}_y"] - frame[f"{center}_y"],
        ]
    )
    denominator = np.linalg.norm(first_vector, axis=1) * np.linalg.norm(third_vector, axis=1)
    cosine = np.divide(
        np.sum(first_vector * third_vector, axis=1),
        denominator,
        out=np.ones(len(frame), dtype=float),
        where=denominator > 1e-12,
    )
    return pd.Series(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), index=frame.index)


def build_compact_signals(normalized_pose: pd.DataFrame, fs: int = FS) -> pd.DataFrame:
    missing = sorted(set(COORD_COLUMNS) - set(normalized_pose.columns))
    if missing:
        raise ValueError(f"Normalized pose is missing coordinates: {missing}")
    out = normalized_pose[COORD_COLUMNS].copy()

    for name, first, second in BONES:
        dx = normalized_pose[f"{second}_x"] - normalized_pose[f"{first}_x"]
        dy = normalized_pose[f"{second}_y"] - normalized_pose[f"{first}_y"]
        out[f"{name}_dx"] = dx
        out[f"{name}_dy"] = dy
        out[f"{name}_length"] = np.hypot(dx, dy)

    for name, first, center, third in ANGLES:
        out[name] = _joint_angle(normalized_pose, first, center, third)

    for joint in JOINTS:
        dx = normalized_pose[f"{joint}_x"].diff().fillna(0) * fs
        dy = normalized_pose[f"{joint}_y"].diff().fillna(0) * fs
        speed = np.hypot(dx, dy)
        out[f"{joint}_speed"] = speed
        out[f"{joint}_acceleration"] = pd.Series(speed, index=out.index).diff().abs().fillna(0) * fs

    distance_pairs = (
        ("left_wrist_to_nose", "left_wrist", "nose"),
        ("right_wrist_to_nose", "right_wrist", "nose"),
        ("left_wrist_to_left_hip", "left_wrist", "left_hip"),
        ("right_wrist_to_right_hip", "right_wrist", "right_hip"),
        ("left_wrist_to_right_wrist", "left_wrist", "right_wrist"),
        ("left_ankle_to_right_ankle", "left_ankle", "right_ankle"),
    )
    for name, first, second in distance_pairs:
        out[name] = _distance(normalized_pose, first, second)

    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def _summarize_window(window: pd.DataFrame) -> dict[str, float]:
    values = window.to_numpy(dtype=float)
    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    centered_time = np.arange(len(window), dtype=float) - (len(window) - 1) / 2
    denominator = float(np.dot(centered_time, centered_time))
    slopes = centered_time @ values / denominator if denominator > 0 else np.zeros(values.shape[1])
    summaries = {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "median": median,
        "iqr": q75 - q25,
        "range": values.max(axis=0) - values.min(axis=0),
        "slope": slopes,
        "energy": np.mean(values**2, axis=0),
        "q10": q10,
        "q90": q90,
    }
    return {
        f"{column}__{summary_name}": float(summary_values[column_index])
        for summary_name, summary_values in summaries.items()
        for column_index, column in enumerate(window.columns)
    }


def extract_compact_windows(
    df: pd.DataFrame,
    subject_id: int,
    window_size: int,
    stride: int,
    majority_threshold: float = 0.70,
    labeled: bool = True,
    fs: int = FS,
) -> WindowFeatures:
    if window_size <= 1 or stride <= 0:
        raise ValueError("window_size must exceed one and stride must be positive")
    prepared = prepare_pose_frame(df)
    if labeled and "Action Label" not in prepared.columns:
        raise ValueError("Labeled extraction requires an 'Action Label' column")
    normalized = normalize_for_experiment(prepared, "C")
    signals = build_compact_signals(normalized, fs=fs)

    valid_starts: list[int] = []
    labels: list[str | None] = []
    for start in window_starts(len(prepared), window_size, stride, cover_tail=not labeled):
        label = None
        if labeled:
            label = majority_label(
                prepared["Action Label"].iloc[start : start + window_size],
                threshold=majority_threshold,
            )
            if label is None:
                continue
        valid_starts.append(start)
        labels.append(label)
    if not valid_starts:
        raise ValueError(f"Subject {subject_id} produced no compact windows")

    rows = [
        _summarize_window(signals.iloc[start : min(start + window_size, len(signals))])
        for start in valid_starts
    ]
    features = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return WindowFeatures(
        x=features,
        y=pd.Series(labels, name="Action Label", dtype="string" if labeled else "object"),
        meta=_window_metadata(prepared, subject_id, valid_starts, window_size),
    )


def subsample_compact_windows(
    windows: WindowFeatures,
    stride: int,
    n_frames: int,
    cover_tail: bool,
) -> WindowFeatures:
    if stride <= 0 or n_frames <= 0:
        raise ValueError("stride and n_frames must be positive")
    if windows.meta.empty:
        raise ValueError("Cannot subsample empty compact windows")
    starts = windows.meta["start_pos"].to_numpy(dtype=int)
    ends = windows.meta["end_pos"].to_numpy(dtype=int)
    window_sizes = ends - starts
    if (window_sizes <= 0).any():
        raise ValueError("Compact window metadata contains an empty interval")
    window_size = int(window_sizes.max())
    keep = starts % stride == 0
    if cover_tail:
        final_start = max(0, n_frames - window_size)
        keep |= starts == final_start
    indices = np.flatnonzero(keep)
    if len(indices) == 0:
        raise ValueError("Requested stride produced no compact windows")
    return WindowFeatures(
        x=windows.x.iloc[indices].reset_index(drop=True),
        y=windows.y.iloc[indices].reset_index(drop=True),
        meta=windows.meta.iloc[indices].reset_index(drop=True),
    )


def label_compact_windows(
    windows: WindowFeatures,
    frame_labels: Sequence[str],
    majority_threshold: float,
) -> WindowFeatures:
    if not 0 < majority_threshold <= 1:
        raise ValueError("majority_threshold must be in (0, 1]")
    labels = np.asarray(frame_labels, dtype=object)
    selected_indices: list[int] = []
    selected_labels: list[str] = []
    for index, row in enumerate(windows.meta[["start_pos", "end_pos"]].itertuples(index=False)):
        start = max(0, int(row.start_pos))
        end = min(len(labels), int(row.end_pos))
        if start >= end:
            raise ValueError(f"Compact window {index} has an empty label interval")
        label = majority_label(pd.Series(labels[start:end]), threshold=majority_threshold)
        if label is not None:
            selected_indices.append(index)
            selected_labels.append(label)
    if not selected_indices:
        raise ValueError("No compact windows passed the majority-label threshold")
    return WindowFeatures(
        x=windows.x.iloc[selected_indices].reset_index(drop=True),
        y=pd.Series(selected_labels, name="Action Label", dtype="string"),
        meta=windows.meta.iloc[selected_indices].reset_index(drop=True),
    )


def compact_cache_path(
    cache_dir: Path,
    subject_id: int,
    window_size: int,
    stride: int,
    majority_threshold: float,
) -> Path:
    threshold_code = int(round(majority_threshold * 100))
    return cache_dir / (
        f"compact_{COMPACT_SCHEMA_VERSION}_subject_{subject_id}"
        f"_w{window_size}_s{stride}_m{threshold_code:03d}.joblib"
    )


def fuse_scale_probabilities(
    probabilities_by_scale: dict[int, np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    weight_tuple = tuple(float(value) for value in weights)
    if weight_tuple not in MULTISCALE_FUSION_WEIGHTS:
        raise ValueError("weights are not in the declared fusion grid")
    if set(probabilities_by_scale) != set(MULTISCALE_WINDOWS):
        raise ValueError(f"probabilities must contain scales {MULTISCALE_WINDOWS}")
    arrays = [np.asarray(probabilities_by_scale[scale], dtype=float) for scale in MULTISCALE_WINDOWS]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("All scale probability arrays must have the same shape")
    if arrays[0].ndim != 2 or not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Scale probabilities must be finite two-dimensional arrays")
    fused = sum(weight * array for weight, array in zip(weight_tuple, arrays))
    row_sums = fused.sum(axis=1, keepdims=True)
    if (fused < 0).any() or (row_sums <= 0).any():
        raise ValueError("Fused probabilities must be non-negative with positive row sums")
    return fused / row_sums
