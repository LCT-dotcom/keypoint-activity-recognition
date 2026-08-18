from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
import tsfel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


CLASSES = [
    "Attacking",
    "Biting",
    "Eating snacks",
    "Head banging",
    "Sitting quietly",
    "Throwing things",
    "Using phone",
    "Walking",
]

ABNORMAL_CLASSES = {"Attacking", "Biting", "Head banging", "Throwing things"}

JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COORD_COLUMNS = [f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y")]

LABEL_ALIASES = {
    "Throwing": "Throwing things",
    "Throwing object": "Throwing things",
    "Throwing objects": "Throwing things",
    "Biting Nails": "Biting",
    "Biting nails": "Biting",
    "Head-Banging": "Head banging",
    "Sitting Quietly": "Sitting quietly",
    "Eating Snacks": "Eating snacks",
    "Using Phone": "Using phone",
}

FS = 30
WINDOW_SIZE = 150
TRAIN_STRIDE = 75
TEST_STRIDE = 150
MAJORITY_THRESHOLD = 0.70
FEATURE_SCHEMA_VERSION = "v7-compatible-1"


@dataclass
class WindowFeatures:
    x: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


@dataclass
class LosoResult:
    fold_metrics: pd.DataFrame
    fold_confusions: dict[int, pd.DataFrame]
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion: pd.DataFrame
    classification_report: pd.DataFrame
    summary: dict[str, float]


@dataclass
class LosoFoldResult:
    held_out_subject: int
    n_train_windows: int
    n_test_windows: int
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion: pd.DataFrame
    classification_report: pd.DataFrame


@dataclass
class FramePrediction:
    frame_labels: np.ndarray
    confidence: np.ndarray
    window_predictions: np.ndarray
    window_probabilities: np.ndarray
    meta: pd.DataFrame


@dataclass
class OutputPaths:
    filled: Path
    submission: Path


def clean_labels(series: pd.Series) -> pd.Series:
    labels = series.astype("string").str.strip().replace(LABEL_ALIASES)
    return labels.where(labels.isin(CLASSES), "None").fillna("None")


def validate_pose_columns(df: pd.DataFrame) -> None:
    missing = [column for column in COORD_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} pose columns: {missing}")


def prepare_pose_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Pose CSV is empty")
    validate_pose_columns(df)
    out = df.copy()
    out[COORD_COLUMNS] = out[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out[COORD_COLUMNS] = out[COORD_COLUMNS].interpolate(limit_direction="both").ffill().bfill()
    if out[COORD_COLUMNS].isna().any().any():
        bad = out[COORD_COLUMNS].columns[out[COORD_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Pose columns contain only missing values: {bad}")
    if "Action Label" in out.columns:
        out["Action Label"] = clean_labels(out["Action Label"])
    return out


def _distance_xy(x1: pd.Series, y1: pd.Series, x2: pd.Series, y2: pd.Series) -> pd.Series:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def pose_normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hip_mid_x = (out["left_hip_x"] + out["right_hip_x"]) / 2
    hip_mid_y = (out["left_hip_y"] + out["right_hip_y"]) / 2
    shoulder_mid_x = (out["left_shoulder_x"] + out["right_shoulder_x"]) / 2
    shoulder_mid_y = (out["left_shoulder_y"] + out["right_shoulder_y"]) / 2

    torso = _distance_xy(shoulder_mid_x, shoulder_mid_y, hip_mid_x, hip_mid_y)
    shoulder_width = _distance_xy(
        out["left_shoulder_x"],
        out["left_shoulder_y"],
        out["right_shoulder_x"],
        out["right_shoulder_y"],
    )
    scale = shoulder_width.where(shoulder_width > 1e-6, torso)
    scale = scale.replace([np.inf, -np.inf], np.nan)
    finite_scale = scale.dropna()
    fallback = float(finite_scale.median()) if not finite_scale.empty else 1.0
    scale = scale.fillna(fallback).clip(lower=1e-6)

    for column in (c for c in COORD_COLUMNS if c.endswith("_x")):
        out[column] = (out[column] - hip_mid_x) / scale
    for column in (c for c in COORD_COLUMNS if c.endswith("_y")):
        out[column] = (out[column] - hip_mid_y) / scale
    return out


def majority_label(labels: Iterable[str], threshold: float = 0.70) -> str | None:
    values = list(labels)
    valid = [label for label in values if label in CLASSES]
    if not values or not valid:
        return None
    label, count = Counter(valid).most_common(1)[0]
    return label if count / len(values) >= threshold else None


def _joint_angle(df: pd.DataFrame, a: str, b: str, c: str) -> pd.Series:
    ba = np.stack([df[f"{a}_x"] - df[f"{b}_x"], df[f"{a}_y"] - df[f"{b}_y"]], axis=1)
    bc = np.stack([df[f"{c}_x"] - df[f"{b}_x"], df[f"{c}_y"] - df[f"{b}_y"]], axis=1)
    denominator = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    cosine = np.clip((ba * bc).sum(axis=1) / denominator, -1.0, 1.0)
    return pd.Series(np.degrees(np.arccos(cosine)), index=df.index)


def _point_velocity(df: pd.DataFrame, point: str) -> pd.Series:
    return np.sqrt(df[f"{point}_x"].diff().pow(2) + df[f"{point}_y"].diff().pow(2)).fillna(0)


def _point_jerk(df: pd.DataFrame, point: str) -> pd.Series:
    return (
        df[f"{point}_x"].diff().diff().abs()
        + df[f"{point}_y"].diff().diff().abs()
    ).fillna(0)


def add_pose_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    distance_pairs = {
        "dist_shoulders": ("left_shoulder", "right_shoulder"),
        "dist_hips": ("left_hip", "right_hip"),
        "dist_knees": ("left_knee", "right_knee"),
        "dist_ankles": ("left_ankle", "right_ankle"),
        "dist_wrist_to_hip": ("right_wrist", "right_hip"),
        "dist_lw_nose": ("left_wrist", "nose"),
        "dist_rw_nose": ("right_wrist", "nose"),
        "dist_lw_ear": ("left_wrist", "left_ear"),
        "dist_rw_ear": ("right_wrist", "right_ear"),
        "dist_la_lhip": ("left_ankle", "left_hip"),
        "dist_ra_rhip": ("right_ankle", "right_hip"),
    }
    for name, (point_a, point_b) in distance_pairs.items():
        out[name] = _distance_xy(
            out[f"{point_a}_x"], out[f"{point_a}_y"], out[f"{point_b}_x"], out[f"{point_b}_y"]
        )

    angle_definitions = {
        "angle_elbow_l": ("left_shoulder", "left_elbow", "left_wrist"),
        "angle_elbow_r": ("right_shoulder", "right_elbow", "right_wrist"),
        "angle_knee_l": ("left_hip", "left_knee", "left_ankle"),
        "angle_knee_r": ("right_hip", "right_knee", "right_ankle"),
        "angle_hip_l": ("left_shoulder", "left_hip", "left_knee"),
        "angle_hip_r": ("right_shoulder", "right_hip", "right_knee"),
    }
    for name, points in angle_definitions.items():
        out[name] = _joint_angle(out, *points)

    out["angle_shoulder_tilt"] = np.degrees(
        np.arctan2(
            out["right_shoulder_y"] - out["left_shoulder_y"],
            out["right_shoulder_x"] - out["left_shoulder_x"],
        )
    )
    shoulder_mid_x = (out["left_shoulder_x"] + out["right_shoulder_x"]) / 2
    shoulder_mid_y = (out["left_shoulder_y"] + out["right_shoulder_y"]) / 2
    hip_mid_x = (out["left_hip_x"] + out["right_hip_x"]) / 2
    hip_mid_y = (out["left_hip_y"] + out["right_hip_y"]) / 2
    out["angle_torso"] = np.degrees(
        np.arctan2(shoulder_mid_y - hip_mid_y, shoulder_mid_x - hip_mid_x)
    )

    floor_y = out[["left_ankle_y", "right_ankle_y"]].max(axis=1)
    shoulder_scale = out["dist_shoulders"].clip(lower=1e-6)
    out["dist_lw_floor_norm"] = (floor_y - out["left_wrist_y"]) / shoulder_scale
    out["dist_rw_floor_norm"] = (floor_y - out["right_wrist_y"]) / shoulder_scale

    out["com_x"] = out[["left_hip_x", "right_hip_x", "left_shoulder_x", "right_shoulder_x"]].mean(axis=1)
    out["com_y"] = out[["left_hip_y", "right_hip_y", "left_shoulder_y", "right_shoulder_y"]].mean(axis=1)
    out["pelvis_y"] = hip_mid_y
    out["head_vertical"] = out["nose_y"]
    out["wrist_y_diff"] = (out["left_wrist_y"] - out["right_wrist_y"]).abs()

    for point in ("nose", "left_wrist", "right_wrist", "left_ankle", "right_ankle", "com"):
        out[f"vel_{point}"] = _point_velocity(out, point)
    out["vel_dist_lw_nose"] = out["dist_lw_nose"].diff().abs().fillna(0)
    out["vel_dist_rw_nose"] = out["dist_rw_nose"].diff().abs().fillna(0)
    out["vel_head_nod"] = out["nose_y"].diff().abs().fillna(0)
    out["vel_knee_angle"] = ((out["angle_knee_l"] + out["angle_knee_r"]) / 2).diff().abs().fillna(0)
    out["vel_torso_angle"] = out["angle_torso"].diff().abs().fillna(0)
    out["vel_shoulder_tilt"] = out["angle_shoulder_tilt"].diff().abs().fillna(0)

    out["left_wrist_jerk"] = _point_jerk(out, "left_wrist")
    out["right_wrist_jerk"] = _point_jerk(out, "right_wrist")
    out["jerk_nose_y"] = out["nose_y"].diff().diff().abs().fillna(0)
    out["jerk_com"] = out["vel_com"].diff().abs().fillna(0)
    out["accel_left_wrist"] = out["vel_left_wrist"].diff().abs().fillna(0)
    out["accel_right_wrist"] = out["vel_right_wrist"].diff().abs().fillna(0)
    out["delta_knee_l_y"] = out["left_knee_y"].diff().fillna(0)
    out["delta_knee_r_y"] = out["right_knee_y"].diff().fillna(0)
    out["duck_proxy"] = out[["delta_knee_l_y", "delta_knee_r_y"]].min(axis=1).abs()
    out["max_frame_jerk_rw"] = out[["left_wrist_jerk", "right_wrist_jerk"]].max(axis=1)

    out["min_dist_wrist_head"] = out[["dist_lw_nose", "dist_rw_nose"]].min(axis=1)
    out["total_movement"] = (
        out["vel_dist_lw_nose"]
        + out["vel_dist_rw_nose"]
        + out["vel_head_nod"]
        + out["left_wrist_jerk"]
        + out["right_wrist_jerk"]
    )
    out["hand_near_head"] = (out["min_dist_wrist_head"] < 0.10).astype(float)
    out["micro_bite"] = (
        out[["vel_dist_lw_nose", "vel_dist_rw_nose"]].gt(0.01)
        & out[["vel_dist_lw_nose", "vel_dist_rw_nose"]].lt(0.05)
    ).any(axis=1).astype(float)
    out["strong_bite"] = out[["vel_dist_lw_nose", "vel_dist_rw_nose"]].gt(0.10).any(axis=1).astype(float)
    out["knee_flex"] = ((out["angle_knee_l"] < 140) & (out["angle_knee_r"] < 140)).astype(float)
    out["knee_extend"] = ((out["angle_knee_l"] > 170) & (out["angle_knee_r"] > 170)).astype(float)
    out["elbow_flex"] = out[["angle_elbow_l", "angle_elbow_r"]].min(axis=1).lt(100).astype(float)
    out["static_wrist"] = (
        (out["vel_left_wrist"] < 0.005) & (out["vel_right_wrist"] < 0.005)
    ).astype(float)
    out["low_motion"] = out["total_movement"].lt(0.01).astype(float)
    out["ground_mask_l"] = out["dist_lw_floor_norm"].lt(0.15).astype(float)
    out["ground_mask_r"] = out["dist_rw_floor_norm"].lt(0.15).astype(float)
    out["both_on_ground"] = (out["ground_mask_l"] * out["ground_mask_r"]).astype(float)
    out["ratio_rwjerk_comjerk"] = (
        out["right_wrist_jerk"] / (out["jerk_com"] + 1e-4)
    ).clip(upper=1000)

    signal_columns = (
        list(distance_pairs)
        + ["dist_lw_floor_norm", "dist_rw_floor_norm"]
        + list(angle_definitions)
        + [
            "angle_shoulder_tilt",
            "angle_torso",
            "com_x",
            "com_y",
            "pelvis_y",
            "head_vertical",
            "wrist_y_diff",
            "vel_dist_lw_nose",
            "vel_dist_rw_nose",
            "vel_head_nod",
            "vel_left_wrist",
            "vel_right_wrist",
            "vel_left_ankle",
            "vel_right_ankle",
            "vel_com",
            "vel_knee_angle",
            "vel_torso_angle",
            "vel_shoulder_tilt",
            "left_wrist_jerk",
            "right_wrist_jerk",
            "jerk_nose_y",
            "jerk_com",
            "accel_left_wrist",
            "accel_right_wrist",
            "delta_knee_l_y",
            "delta_knee_r_y",
            "duck_proxy",
            "max_frame_jerk_rw",
            "min_dist_wrist_head",
            "total_movement",
            "hand_near_head",
            "micro_bite",
            "strong_bite",
            "knee_flex",
            "knee_extend",
            "elbow_flex",
            "static_wrist",
            "low_motion",
            "ground_mask_l",
            "ground_mask_r",
            "both_on_ground",
            "ratio_rwjerk_comjerk",
        ]
    )
    out[signal_columns] = out[signal_columns].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    return out, signal_columns


def make_tsfel_config() -> dict:
    full_config = tsfel.get_features_by_domain(["statistical", "temporal", "spectral"])
    whitelist = {
        "statistical": {
            "Absolute energy",
            "Interquartile range",
            "Max",
            "Mean",
            "Median",
            "Min",
            "Peak to peak distance",
            "Root mean square",
            "Standard deviation",
            "Variance",
            "Mean absolute deviation",
            "Median absolute deviation",
        },
        "temporal": {
            "Area under the curve",
            "Autocorrelation",
            "Mean absolute diff",
            "Slope",
            "Zero crossing rate",
            "Median absolute diff",
            "Negative turning points",
            "Neighbourhood peaks",
            "Positive turning points",
            "Signal distance",
            "Sum absolute diff",
        },
        "spectral": {
            "Fundamental frequency",
            "Max power spectrum",
            "Maximum frequency",
            "Median frequency",
            "Spectral centroid",
            "Spectral entropy",
            "Spectral roll-off",
            "Spectral spread",
        },
    }
    return {
        domain: {
            name: specification
            for name, specification in full_config[domain].items()
            if name in whitelist[domain]
        }
        for domain in ("statistical", "temporal", "spectral")
    }


def window_starts(length: int, window_size: int, stride: int, cover_tail: bool = False) -> list[int]:
    if length <= 0:
        return []
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    tail_start = length - window_size
    if cover_tail and starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _padded_window(frame: pd.DataFrame, start: int, window_size: int) -> pd.DataFrame:
    window = frame.iloc[start : start + window_size].reset_index(drop=True)
    if len(window) == window_size:
        return window
    if window.empty:
        raise ValueError("Cannot extract a window from an empty DataFrame")
    padding = pd.DataFrame(
        np.repeat(window.iloc[[-1]].to_numpy(), window_size - len(window), axis=0),
        columns=window.columns,
    )
    return pd.concat([window, padding], ignore_index=True)


def _extract_tsfel_features(
    signals: pd.DataFrame,
    starts: list[int],
    window_size: int,
    config: dict,
    fs: int,
) -> pd.DataFrame:
    if not starts:
        return pd.DataFrame()
    windows = [_padded_window(signals, start, window_size) for start in starts]
    features = tsfel.time_series_features_extractor(
        config,
        windows,
        fs=fs,
        verbose=0,
        n_jobs=None,
    )
    features = features.apply(pd.to_numeric, errors="coerce")
    return features.mask(np.isinf(features))


def _window_metadata(df: pd.DataFrame, subject_id: int, starts: list[int], window_size: int) -> pd.DataFrame:
    rows = []
    for start in starts:
        end = min(start + window_size, len(df))
        rows.append(
            {
                "subject_id": subject_id,
                "start_pos": start,
                "end_pos": end,
                "frame_start": int(df["frame_id"].iloc[start]) if "frame_id" in df.columns else start,
                "frame_end": int(df["frame_id"].iloc[end - 1]) if "frame_id" in df.columns else end - 1,
            }
        )
    return pd.DataFrame(rows)


def extract_labeled_windows(
    df: pd.DataFrame,
    subject_id: int,
    config: dict,
    window_size: int = WINDOW_SIZE,
    stride: int = TRAIN_STRIDE,
    majority_threshold: float = MAJORITY_THRESHOLD,
    fs: int = FS,
) -> WindowFeatures:
    prepared = prepare_pose_frame(df)
    if "Action Label" not in prepared.columns:
        raise ValueError("Training CSV must contain an 'Action Label' column")
    normalized = pose_normalize(prepared)
    enriched, signal_columns = add_pose_signals(normalized)

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

    x = _extract_tsfel_features(enriched[signal_columns], valid_starts, window_size, config, fs)
    return WindowFeatures(
        x=x,
        y=pd.Series(labels, name="Action Label", dtype="string"),
        meta=_window_metadata(prepared, subject_id, valid_starts, window_size),
    )


def extract_unlabeled_windows(
    df: pd.DataFrame,
    subject_id: int,
    config: dict,
    window_size: int = WINDOW_SIZE,
    stride: int = TEST_STRIDE,
    fs: int = FS,
) -> WindowFeatures:
    prepared = prepare_pose_frame(df)
    normalized = pose_normalize(prepared)
    enriched, signal_columns = add_pose_signals(normalized)
    starts = window_starts(len(prepared), window_size, stride, cover_tail=True)
    x = _extract_tsfel_features(enriched[signal_columns], starts, window_size, config, fs)
    return WindowFeatures(
        x=x,
        y=pd.Series([None] * len(starts), name="Action Label", dtype="object"),
        meta=_window_metadata(prepared, subject_id, starts, window_size),
    )


def concatenate_window_features(items: list[WindowFeatures]) -> WindowFeatures:
    if not items:
        raise ValueError("At least one WindowFeatures object is required")
    return WindowFeatures(
        x=pd.concat([item.x for item in items], ignore_index=True),
        y=pd.concat([item.y for item in items], ignore_index=True),
        meta=pd.concat([item.meta for item in items], ignore_index=True),
    )


def select_nonoverlapping_windows(
    windows: WindowFeatures,
    stride: int = TEST_STRIDE,
) -> WindowFeatures:
    if "start_pos" not in windows.meta.columns:
        raise ValueError("Window metadata must contain start_pos for non-overlapping evaluation")
    mask = windows.meta["start_pos"].astype(int).mod(stride).eq(0).to_numpy()
    if not mask.any():
        raise ValueError("No held-out windows match the requested evaluation stride")
    return WindowFeatures(
        x=windows.x.loc[mask].reset_index(drop=True),
        y=windows.y.loc[mask].reset_index(drop=True),
        meta=windows.meta.loc[mask].reset_index(drop=True),
    )


def make_estimator(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold(threshold=1e-10)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_estimator(windows: WindowFeatures, random_state: int = 42) -> Pipeline:
    if windows.x.empty or windows.y.empty:
        raise ValueError("Cannot fit estimator with no windows")
    model = make_estimator(random_state=random_state)
    class_counts = windows.y.value_counts()
    validation_rows = int(np.ceil(len(windows.y) * 0.12))
    if class_counts.min() < 2 or validation_rows < len(class_counts):
        model.set_params(classifier__early_stopping=False)
    weights = compute_sample_weight(class_weight="balanced", y=windows.y)
    model.fit(windows.x, windows.y, classifier__sample_weight=weights)
    return model


def evaluate_predictions(y_true: Iterable[str], y_pred: Iterable[str]) -> dict:
    true = np.asarray(list(y_true), dtype=object)
    predicted = np.asarray(list(y_pred), dtype=object)
    if len(true) != len(predicted) or len(true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and have equal length")

    true_abnormal = np.asarray([label in ABNORMAL_CLASSES for label in true], dtype=int)
    predicted_abnormal = np.asarray([label in ABNORMAL_CLASSES for label in predicted], dtype=int)
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "abnormal_f1": float(f1_score(true_abnormal, predicted_abnormal, zero_division=0)),
        "abnormal_precision": float(precision_score(true_abnormal, predicted_abnormal, zero_division=0)),
        "abnormal_recall": float(recall_score(true_abnormal, predicted_abnormal, zero_division=0)),
        "classification_report": classification_report(
            true,
            predicted,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def run_loso_fold(
    subject_windows: dict[int, WindowFeatures],
    held_out_subject: int,
    random_state: int = 42,
) -> LosoFoldResult:
    subject_ids = sorted(subject_windows)
    if len(subject_ids) < 2:
        raise ValueError("LOSO requires at least two participants")
    if held_out_subject not in subject_windows:
        raise ValueError(f"Held-out subject {held_out_subject} is not available")

    train = concatenate_window_features(
        [subject_windows[subject] for subject in subject_ids if subject != held_out_subject]
    )
    test = select_nonoverlapping_windows(subject_windows[held_out_subject])
    model = fit_estimator(train, random_state=random_state + held_out_subject)
    predicted = model.predict(test.x)
    evaluated = evaluate_predictions(test.y, predicted)
    metrics = {
        key: value for key, value in evaluated.items() if key != "classification_report"
    }
    fold_confusion = pd.DataFrame(
        confusion_matrix(test.y, predicted, labels=CLASSES),
        index=CLASSES,
        columns=CLASSES,
    )
    return LosoFoldResult(
        held_out_subject=held_out_subject,
        n_train_windows=len(train.y),
        n_test_windows=len(test.y),
        metrics=metrics,
        y_true=test.y.astype(str).to_numpy(dtype=object),
        y_pred=np.asarray(predicted, dtype=object),
        confusion=fold_confusion,
        classification_report=pd.DataFrame(evaluated["classification_report"]).transpose(),
    )


def combine_loso_folds(folds: Iterable[LosoFoldResult]) -> LosoResult:
    ordered_folds = sorted(list(folds), key=lambda fold: fold.held_out_subject)
    if not ordered_folds:
        raise ValueError("At least one LOSO fold is required")
    subject_ids = [fold.held_out_subject for fold in ordered_folds]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("Each held-out subject may appear only once")

    fold_rows = []
    all_true: list[str] = []
    all_predicted: list[str] = []
    for fold in ordered_folds:
        fold_rows.append(
            {
                "held_out_subject": fold.held_out_subject,
                "n_train_windows": fold.n_train_windows,
                "n_test_windows": fold.n_test_windows,
                "accuracy": fold.metrics["accuracy"],
                "macro_f1": fold.metrics["macro_f1"],
                "abnormal_f1": fold.metrics["abnormal_f1"],
                "abnormal_precision": fold.metrics["abnormal_precision"],
                "abnormal_recall": fold.metrics["abnormal_recall"],
            }
        )
        all_true.extend(fold.y_true.astype(str).tolist())
        all_predicted.extend(fold.y_pred.astype(str).tolist())

    fold_metrics = pd.DataFrame(fold_rows)
    combined = evaluate_predictions(all_true, all_predicted)
    confusion = pd.DataFrame(
        confusion_matrix(all_true, all_predicted, labels=CLASSES),
        index=CLASSES,
        columns=CLASSES,
    )
    report = pd.DataFrame(combined["classification_report"]).transpose()
    summary = {
        "mean_fold_accuracy": float(fold_metrics["accuracy"].mean()),
        "std_fold_accuracy": float(fold_metrics["accuracy"].std(ddof=0)),
        "mean_fold_macro_f1": float(fold_metrics["macro_f1"].mean()),
        "mean_fold_abnormal_f1": float(fold_metrics["abnormal_f1"].mean()),
        "pooled_accuracy": combined["accuracy"],
        "pooled_macro_f1": combined["macro_f1"],
        "pooled_abnormal_f1": combined["abnormal_f1"],
    }
    return LosoResult(
        fold_metrics=fold_metrics,
        fold_confusions={fold.held_out_subject: fold.confusion for fold in ordered_folds},
        y_true=np.asarray(all_true, dtype=object),
        y_pred=np.asarray(all_predicted, dtype=object),
        confusion=confusion,
        classification_report=report,
        summary=summary,
    )


def run_loso(subject_windows: dict[int, WindowFeatures], random_state: int = 42) -> LosoResult:
    subject_ids = sorted(subject_windows)
    if len(subject_ids) < 2:
        raise ValueError("LOSO requires at least two participants")
    folds = [
        run_loso_fold(subject_windows, held_out, random_state=random_state)
        for held_out in subject_ids
    ]
    return combine_loso_folds(folds)


def build_artifact(model: Pipeline, feature_columns: Iterable[str], metadata: dict) -> dict:
    return {
        "format_version": 2,
        "architecture": "TSFEL+HistGradientBoosting",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model": model,
        "feature_columns": list(feature_columns),
        "classes": CLASSES.copy(),
        "abnormal_classes": sorted(ABNORMAL_CLASSES),
        "coordinate_columns": COORD_COLUMNS.copy(),
        "label_aliases": LABEL_ALIASES.copy(),
        "tsfel_config": make_tsfel_config(),
        "preprocessing": {
            "fps": FS,
            "window_size": WINDOW_SIZE,
            "train_stride": TRAIN_STRIDE,
            "test_stride": TEST_STRIDE,
            "majority_threshold": MAJORITY_THRESHOLD,
            "normalization": "hip midpoint center; shoulder-width scale; torso-length fallback",
        },
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "tsfel": getattr(tsfel, "__version__", "0.2.0"),
            "joblib": joblib.__version__,
        },
        "metadata": dict(metadata),
    }


def save_artifact(path: Path | str, artifact: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)


def load_artifact(path: Path | str) -> dict:
    artifact = joblib.load(Path(path))
    required = {
        "model",
        "feature_columns",
        "classes",
        "tsfel_config",
        "preprocessing",
        "feature_schema_version",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Incompatible artifact; missing keys: {missing}")
    if artifact["classes"] != CLASSES:
        raise ValueError("Artifact class order does not match the pipeline")
    if artifact["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Artifact feature schema does not match the pipeline")
    return artifact


def predict_frame_labels(
    df: pd.DataFrame,
    artifact: dict,
    participant_id: int,
) -> FramePrediction:
    required = {"model", "feature_columns", "classes", "tsfel_config", "preprocessing"}
    missing_keys = sorted(required - set(artifact))
    if missing_keys:
        raise ValueError(f"Incompatible artifact; missing keys: {missing_keys}")
    if artifact["classes"] != CLASSES:
        raise ValueError("Artifact class order does not match the pipeline")

    preprocessing = artifact["preprocessing"]
    windows = extract_unlabeled_windows(
        df,
        subject_id=participant_id,
        config=artifact["tsfel_config"],
        window_size=int(preprocessing["window_size"]),
        stride=int(preprocessing["test_stride"]),
        fs=int(preprocessing["fps"]),
    )
    expected_columns = list(artifact["feature_columns"])
    missing_features = sorted(set(expected_columns) - set(windows.x.columns))
    if missing_features:
        raise ValueError(
            f"Extracted test features are incompatible with the model; "
            f"missing {len(missing_features)} columns"
        )
    x = windows.x.reindex(columns=expected_columns)

    model = artifact["model"]
    raw_probabilities = model.predict_proba(x)
    model_classes = [str(label) for label in model.classes_]
    unknown_classes = sorted(set(model_classes) - set(CLASSES))
    if unknown_classes:
        raise ValueError(f"Model predicts unsupported classes: {unknown_classes}")

    probabilities = np.zeros((len(x), len(CLASSES)), dtype=float)
    for model_index, label in enumerate(model_classes):
        probabilities[:, CLASSES.index(label)] = raw_probabilities[:, model_index]
    window_predictions = np.asarray(CLASSES, dtype=object)[probabilities.argmax(axis=1)]

    frame_scores = np.zeros((len(df), len(CLASSES)), dtype=float)
    frame_votes = np.zeros(len(df), dtype=int)
    for window_index, row in windows.meta.reset_index(drop=True).iterrows():
        start = int(row["start_pos"])
        end = int(row["end_pos"])
        frame_scores[start:end] += probabilities[window_index]
        frame_votes[start:end] += 1
    if np.any(frame_votes == 0):
        uncovered = np.flatnonzero(frame_votes == 0)
        raise RuntimeError(f"Windowing left {len(uncovered)} test rows without a prediction")

    frame_probabilities = frame_scores / frame_votes[:, None]
    label_indices = frame_probabilities.argmax(axis=1)
    return FramePrediction(
        frame_labels=np.asarray(CLASSES, dtype=object)[label_indices],
        confidence=frame_probabilities[np.arange(len(df)), label_indices],
        window_predictions=window_predictions,
        window_probabilities=probabilities,
        meta=windows.meta.copy(),
    )


def write_prediction_outputs(
    output_dir: Path | str,
    source_df: pd.DataFrame,
    prediction: FramePrediction,
    participant_id: int,
) -> OutputPaths:
    if len(source_df) != len(prediction.frame_labels) or len(source_df) != len(prediction.confidence):
        raise ValueError("Prediction length must match the input CSV row count")
    invalid = sorted(set(prediction.frame_labels) - set(CLASSES))
    if invalid:
        raise ValueError(f"Predictions contain unsupported classes: {invalid}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    filled_path = destination / "test_data_keypoint_shared_predicted.csv"
    submission_path = destination / "submission_tsfel_histgb.csv"

    filled = source_df.copy()
    filled["predicted_label"] = prediction.frame_labels
    filled["prediction_confidence"] = prediction.confidence
    filled.to_csv(filled_path, index=False)

    if "timestamp" in source_df.columns:
        timestamp = source_df["timestamp"].to_numpy()
    elif "frame_id" in source_df.columns:
        timestamp = source_df["frame_id"].to_numpy()
    else:
        timestamp = np.arange(len(source_df))
    submission = pd.DataFrame(
        {
            "participant_id": participant_id,
            "timestamp": timestamp,
            "predicted_label": prediction.frame_labels,
        }
    )
    submission.to_csv(submission_path, index=False)
    return OutputPaths(filled=filled_path, submission=submission_path)
