from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from phase_d_classical import ClassicalConfig, TrainingOnlyFeatureSelector, fit_classical_model
from phase_d_evaluation import aggregate_window_probabilities
from phase_d_multiscale import (
    MULTISCALE_FUSION_WEIGHTS,
    MULTISCALE_WINDOWS,
    fuse_scale_probabilities,
    label_compact_windows,
    subsample_compact_windows,
)
from tsfel_histgb_pipeline import CLASSES, WindowFeatures, concatenate_window_features, evaluate_predictions


D2_SEARCH_VERSION = "phase-d2-pair-search-v1"
D2_THRESHOLDS = (0.70, 0.85)
D2_INFERENCE_STRIDES = (15, 30)


@dataclass(frozen=True)
class ScaleSelection:
    scale: int
    candidate_index: int
    majority_threshold: float
    inference_stride: int
    inner_summary: dict[str, float]


def _load_joblib_if_valid(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except (EOFError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_joblib_dump(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def _parse_pair(value: object) -> tuple[int, ...]:
    return tuple(sorted(int(part) for part in str(value).split("_") if part))


def _rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        ["mean_accuracy", "worst_accuracy", "mean_macro_f1", "mean_abnormal_f1"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def select_scale_settings(
    scores: pd.DataFrame,
    subject_ids: Sequence[int],
    outer_held_out_subject: int,
    allowed_candidate_indices: Sequence[int],
) -> tuple[dict[int, ScaleSelection], dict]:
    subjects = tuple(sorted(int(subject) for subject in subject_ids))
    if outer_held_out_subject not in subjects:
        raise ValueError("Outer held-out subject is unavailable")
    allowed = {int(index) for index in allowed_candidate_indices}
    if not allowed:
        raise ValueError("At least one D1 candidate index must be allowed")
    required = {
        "train_subjects",
        "scale",
        "candidate_index",
        "majority_threshold",
        "inference_stride",
        "validation_subject",
        "accuracy",
        "macro_f1",
        "abnormal_f1",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"D2 scores are missing columns: {missing}")

    normalized = scores.copy()
    normalized["train_tuple"] = normalized["train_subjects"].map(_parse_pair)
    for column in ("scale", "candidate_index", "inference_stride", "validation_subject"):
        normalized[column] = normalized[column].astype(int)
    outer_train = tuple(subject for subject in subjects if subject != outer_held_out_subject)
    inner_splits = [
        {
            "train_subjects": list(subject for subject in outer_train if subject != validation),
            "validation_subject": validation,
        }
        for validation in outer_train
    ]
    selected_frames: list[pd.DataFrame] = []
    for split in inner_splits:
        train_tuple = tuple(split["train_subjects"])
        validation = split["validation_subject"]
        selected_frames.append(
            normalized[
                (normalized["train_tuple"] == train_tuple)
                & (normalized["validation_subject"] == validation)
                & normalized["candidate_index"].isin(allowed)
            ]
        )
    inner = pd.concat(selected_frames, ignore_index=True)
    if inner.empty:
        raise ValueError("No D2 inner scores match the requested outer fold")
    expected_per_scale = len(inner_splits) * len(allowed) * len(D2_THRESHOLDS) * len(D2_INFERENCE_STRIDES)
    selections: dict[int, ScaleSelection] = {}
    for scale in MULTISCALE_WINDOWS:
        scale_rows = inner[inner["scale"] == scale]
        if len(scale_rows) != expected_per_scale:
            raise ValueError(
                f"Scale {scale} expected {expected_per_scale} inner scores; found {len(scale_rows)}"
            )
        summary = scale_rows.groupby(
            ["candidate_index", "majority_threshold", "inference_stride"], as_index=False
        ).agg(
            mean_accuracy=("accuracy", "mean"),
            worst_accuracy=("accuracy", "min"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_abnormal_f1=("abnormal_f1", "mean"),
        )
        best = _rank_summary(summary).iloc[0]
        selections[scale] = ScaleSelection(
            scale=scale,
            candidate_index=int(best["candidate_index"]),
            majority_threshold=float(best["majority_threshold"]),
            inference_stride=int(best["inference_stride"]),
            inner_summary={
                "mean_accuracy": float(best["mean_accuracy"]),
                "worst_accuracy": float(best["worst_accuracy"]),
                "mean_macro_f1": float(best["mean_macro_f1"]),
                "mean_abnormal_f1": float(best["mean_abnormal_f1"]),
            },
        )

    leakage_ok = all(
        outer_held_out_subject not in split["train_subjects"]
        and split["validation_subject"] != outer_held_out_subject
        for split in inner_splits
    )
    if not leakage_ok:
        raise ValueError("Outer held-out subject leaked into D2 scale selection")
    return selections, {
        "outer_held_out_subject": outer_held_out_subject,
        "outer_train_subjects": list(outer_train),
        "inner_splits": inner_splits,
        "allowed_candidate_indices": sorted(allowed),
        "leakage_check_passed": leakage_ok,
    }


def top_d1_candidate_indices(inner_metrics: pd.DataFrame, top_k: int = 2) -> list[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    summary = inner_metrics.groupby("candidate_index", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        worst_accuracy=("accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_abnormal_f1=("abnormal_f1", "mean"),
    )
    ranked = _rank_summary(summary)
    return ranked.head(top_k)["candidate_index"].astype(int).tolist()


def select_fusion_weight(
    probabilities_by_split: dict[object, dict[int, np.ndarray]],
    truth_by_split: dict[object, Sequence[str]],
) -> tuple[tuple[float, float, float], pd.DataFrame]:
    if not probabilities_by_split or set(probabilities_by_split) != set(truth_by_split):
        raise ValueError("Fusion probabilities and truth must contain the same non-empty splits")
    rows: list[dict] = []
    valid_classes = set(CLASSES)
    for weight_index, weights in enumerate(MULTISCALE_FUSION_WEIGHTS):
        for split_key, probabilities in probabilities_by_split.items():
            fused = fuse_scale_probabilities(probabilities, weights)
            truth = np.asarray(truth_by_split[split_key], dtype=object)
            if len(truth) != len(fused):
                raise ValueError("Fusion truth and frame probabilities must have equal lengths")
            valid = np.asarray([label in valid_classes for label in truth], dtype=bool)
            predicted = np.asarray(CLASSES, dtype=object)[fused.argmax(axis=1)]
            metrics = evaluate_predictions(truth[valid], predicted[valid])
            rows.append(
                {
                    "weight_index": weight_index,
                    "weights": "_".join(f"{weight:.6g}" for weight in weights),
                    "split": str(split_key),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "abnormal_f1": metrics["abnormal_f1"],
                }
            )
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("weight_index", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        worst_accuracy=("accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_abnormal_f1=("abnormal_f1", "mean"),
    )
    best_index = int(_rank_summary(summary).iloc[0]["weight_index"])
    return tuple(MULTISCALE_FUSION_WEIGHTS[best_index]), metrics


def prepare_d2_window_caches(
    base_windows: dict[int, dict[int, WindowFeatures]],
    frame_truth: dict[int, Sequence[str]],
) -> tuple[dict[tuple[int, int, float], WindowFeatures], dict[tuple[int, int, int], WindowFeatures]]:
    training: dict[tuple[int, int, float], WindowFeatures] = {}
    inference: dict[tuple[int, int, int], WindowFeatures] = {}
    for subject, scale_windows in base_windows.items():
        n_frames = len(frame_truth[subject])
        for scale, base in scale_windows.items():
            train_unlabeled = subsample_compact_windows(
                base, stride=scale // 2, n_frames=n_frames, cover_tail=False
            )
            for threshold in D2_THRESHOLDS:
                training[(subject, scale, threshold)] = label_compact_windows(
                    train_unlabeled, frame_truth[subject], threshold
                )
            for stride in D2_INFERENCE_STRIDES:
                inference[(subject, scale, stride)] = subsample_compact_windows(
                    base, stride=stride, n_frames=n_frames, cover_tail=True
                )
    return training, inference


def _d2_training_signature(
    train_subjects: tuple[int, ...],
    scale: int,
    threshold: float,
    windows: WindowFeatures,
    max_budget: int,
) -> dict:
    columns = json.dumps(list(windows.x.columns), ensure_ascii=True).encode("utf-8")
    return {
        "version": D2_SEARCH_VERSION,
        "train_subjects": list(train_subjects),
        "scale": scale,
        "majority_threshold": threshold,
        "rows": len(windows.x),
        "columns": len(windows.x.columns),
        "columns_sha256": hashlib.sha256(columns).hexdigest(),
        "label_counts": {
            str(label): int(count)
            for label, count in windows.y.value_counts().sort_index().items()
        },
        "max_budget": max_budget,
    }


def d2_selector_path(
    output_dir: Path, train_subjects: tuple[int, ...], scale: int, threshold: float
) -> Path:
    pair = "_".join(map(str, train_subjects))
    return output_dir / f"selector_pair_{pair}_w{scale}_m{int(round(threshold * 100)):03d}.joblib"


def d2_model_path(
    output_dir: Path,
    train_subjects: tuple[int, ...],
    scale: int,
    threshold: float,
    candidate_index: int,
) -> Path:
    pair = "_".join(map(str, train_subjects))
    return output_dir / (
        f"model_pair_{pair}_w{scale}_m{int(round(threshold * 100)):03d}"
        f"_candidate_{candidate_index:02d}.joblib"
    )


def run_resumable_d2_pair_search(
    base_windows: dict[int, dict[int, WindowFeatures]],
    frame_truth: dict[int, Sequence[str]],
    candidates: Sequence[ClassicalConfig],
    allowed_candidates_by_outer: dict[int, Sequence[int]],
    output_dir: Path,
    random_state: int = 42,
    force: bool = False,
    train_pairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    subjects = tuple(sorted(base_windows))
    if len(subjects) != 4 or set(subjects) != set(frame_truth):
        raise ValueError("D2 pair search requires matching data for four development subjects")
    if set(allowed_candidates_by_outer) != set(subjects):
        raise ValueError("Allowed D2 candidates must be declared for every outer subject")
    candidate_list = list(candidates)
    for outer, indices in allowed_candidates_by_outer.items():
        if not indices or any(index < 0 or index >= len(candidate_list) for index in indices):
            raise ValueError(f"Outer subject {outer} has an invalid candidate list")
    if any(set(scale_windows) != set(MULTISCALE_WINDOWS) for scale_windows in base_windows.values()):
        raise ValueError(f"Every subject must contain compact scales {MULTISCALE_WINDOWS}")

    all_pairs = tuple(combinations(subjects, 2))
    selected_pairs = tuple(
        tuple(sorted(map(int, pair))) for pair in (train_pairs if train_pairs is not None else all_pairs)
    )
    if not selected_pairs or len(set(selected_pairs)) != len(selected_pairs):
        raise ValueError("D2 train-pair filter must be non-empty and unique")
    if any(pair not in all_pairs for pair in selected_pairs):
        raise ValueError("D2 train-pair filter contains an unknown subject pair")

    output_dir.mkdir(parents=True, exist_ok=True)
    training, inference = prepare_d2_window_caches(base_windows, frame_truth)
    audit = {
        "version": D2_SEARCH_VERSION,
        "selector_fits": 0,
        "model_fits": 0,
        "cache_hits": 0,
        "completed_models": 0,
        "random_state": random_state,
        "train_pairs": [list(pair) for pair in selected_pairs],
    }
    all_scores: list[pd.DataFrame] = []

    for train_subjects in selected_pairs:
        validation_subjects = tuple(subject for subject in subjects if subject not in train_subjects)
        needed_indices = sorted(
            {
                int(index)
                for outer_subject in validation_subjects
                for index in allowed_candidates_by_outer[outer_subject]
            }
        )
        max_budget = max(candidate_list[index].feature_budget for index in needed_indices)
        for scale in MULTISCALE_WINDOWS:
            for threshold in D2_THRESHOLDS:
                train = concatenate_window_features(
                    [training[(subject, scale, threshold)] for subject in train_subjects]
                )
                signature = _d2_training_signature(
                    train_subjects, scale, threshold, train, max_budget
                )
                selector_file = d2_selector_path(output_dir, train_subjects, scale, threshold)
                selector = None
                if selector_file.exists() and not force:
                    payload = _load_joblib_if_valid(selector_file)
                    if payload is not None and payload.get("signature") == signature:
                        selector = payload.get("selector")
                if selector is None:
                    selector = TrainingOnlyFeatureSelector(
                        feature_budget=max_budget,
                        random_state=random_state + sum(train_subjects) + scale,
                    ).fit(train.x, train.y)
                    _atomic_joblib_dump(
                        {"signature": signature, "selector": selector},
                        selector_file,
                    )
                    audit["selector_fits"] += 1

                for candidate_index in needed_indices:
                    config = candidate_list[candidate_index]
                    model_file = d2_model_path(
                        output_dir, train_subjects, scale, threshold, candidate_index
                    )
                    model_signature = {
                        "training": signature,
                        "candidate_index": candidate_index,
                        "config": config.to_dict(),
                        "random_state": random_state,
                    }
                    cached_scores = None
                    if model_file.exists() and not force:
                        payload = _load_joblib_if_valid(model_file)
                        if payload is not None and payload.get("signature") == model_signature:
                            cached_scores = payload.get("scores")
                    if cached_scores is not None:
                        all_scores.append(cached_scores)
                        audit["cache_hits"] += 1
                        audit["completed_models"] += 1
                        continue

                    fitted = fit_classical_model(
                        train,
                        config=config,
                        random_state=(
                            random_state
                            + candidate_index
                            + sum(train_subjects) * 100
                            + scale
                            + int(threshold * 1000)
                        ),
                        prefit_selector=selector.restrict_budget(config.feature_budget),
                    )
                    rows: list[dict] = []
                    for validation_subject in validation_subjects:
                        truth = np.asarray(frame_truth[validation_subject], dtype=object)
                        valid = np.asarray([label in set(CLASSES) for label in truth], dtype=bool)
                        for stride in D2_INFERENCE_STRIDES:
                            validation = inference[(validation_subject, scale, stride)]
                            probabilities = fitted.predict_proba(validation.x, CLASSES)
                            frame_result = aggregate_window_probabilities(
                                validation.meta,
                                probabilities,
                                n_frames=len(truth),
                                classes=CLASSES,
                            )
                            metrics = evaluate_predictions(truth[valid], frame_result.labels[valid])
                            rows.append(
                                {
                                    "train_subjects": "_".join(map(str, train_subjects)),
                                    "scale": scale,
                                    "candidate_index": candidate_index,
                                    "majority_threshold": threshold,
                                    "inference_stride": stride,
                                    "validation_subject": validation_subject,
                                    "accuracy": metrics["accuracy"],
                                    "macro_f1": metrics["macro_f1"],
                                    "abnormal_f1": metrics["abnormal_f1"],
                                }
                            )
                    model_scores = pd.DataFrame(rows)
                    _atomic_joblib_dump(
                        {"signature": model_signature, "model": fitted, "scores": model_scores},
                        model_file,
                    )
                    all_scores.append(model_scores)
                    audit["model_fits"] += 1
                    audit["completed_models"] += 1

    combined = pd.concat(all_scores, ignore_index=True).sort_values(
        [
            "train_subjects",
            "scale",
            "majority_threshold",
            "candidate_index",
            "validation_subject",
            "inference_stride",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    suffix = "" if train_pairs is None else "_" + "__".join(
        "_".join(map(str, pair)) for pair in selected_pairs
    )
    combined.to_csv(output_dir / f"d2_pair_search_scores{suffix}.csv", index=False)
    (output_dir / f"d2_pair_search_audit{suffix}.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return combined, audit


def reconstruct_inner_scale_probabilities(
    inference_windows: dict[tuple[int, int, int], WindowFeatures],
    frame_truth: dict[int, Sequence[str]],
    selections: dict[int, ScaleSelection],
    selection_audit: dict,
    search_dir: Path,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, np.ndarray]]:
    probabilities_by_split: dict[str, dict[int, np.ndarray]] = {}
    truth_by_split: dict[str, np.ndarray] = {}
    for split in selection_audit["inner_splits"]:
        train_subjects = tuple(sorted(int(subject) for subject in split["train_subjects"]))
        validation_subject = int(split["validation_subject"])
        split_key = f"train_{'_'.join(map(str, train_subjects))}_val_{validation_subject}"
        truth = np.asarray(frame_truth[validation_subject], dtype=object)
        truth_by_split[split_key] = truth
        scale_probabilities: dict[int, np.ndarray] = {}
        for scale, selection in selections.items():
            payload = joblib.load(
                d2_model_path(
                    search_dir,
                    train_subjects,
                    scale,
                    selection.majority_threshold,
                    selection.candidate_index,
                )
            )
            model = payload["model"]
            validation = inference_windows[
                (validation_subject, scale, selection.inference_stride)
            ]
            window_probabilities = model.predict_proba(validation.x, CLASSES)
            frame_result = aggregate_window_probabilities(
                validation.meta,
                window_probabilities,
                n_frames=len(truth),
                classes=CLASSES,
            )
            scale_probabilities[scale] = frame_result.probabilities
        probabilities_by_split[split_key] = scale_probabilities
    return probabilities_by_split, truth_by_split


def fit_d2_outer_models(
    training_windows: dict[tuple[int, int, float], WindowFeatures],
    inference_windows: dict[tuple[int, int, int], WindowFeatures],
    frame_truth: dict[int, Sequence[str]],
    candidates: Sequence[ClassicalConfig],
    outer_held_out_subject: int,
    selections: dict[int, ScaleSelection],
    fusion_weights: Sequence[float],
    random_state: int = 42,
) -> tuple[dict[int, object], object, dict]:
    from phase_d_multiscale import fuse_scale_probabilities

    candidate_list = list(candidates)
    train_subjects = tuple(
        sorted(subject for subject in frame_truth if subject != outer_held_out_subject)
    )
    fitted_by_scale: dict[int, object] = {}
    frame_probabilities: dict[int, np.ndarray] = {}
    scale_audit: dict[str, dict] = {}
    for scale, selection in selections.items():
        config = candidate_list[selection.candidate_index]
        train = concatenate_window_features(
            [
                training_windows[(subject, scale, selection.majority_threshold)]
                for subject in train_subjects
            ]
        )
        selector = TrainingOnlyFeatureSelector(
            feature_budget=config.feature_budget,
            random_state=random_state + outer_held_out_subject + scale,
        ).fit(train.x, train.y)
        fitted = fit_classical_model(
            train,
            config=config,
            random_state=random_state + outer_held_out_subject + scale,
            prefit_selector=selector,
        )
        test = inference_windows[
            (outer_held_out_subject, scale, selection.inference_stride)
        ]
        window_probabilities = fitted.predict_proba(test.x, CLASSES)
        frame_result = aggregate_window_probabilities(
            test.meta,
            window_probabilities,
            n_frames=len(frame_truth[outer_held_out_subject]),
            classes=CLASSES,
        )
        fitted_by_scale[scale] = fitted
        frame_probabilities[scale] = frame_result.probabilities
        scale_audit[str(scale)] = {
            "candidate_index": selection.candidate_index,
            "config": config.to_dict(),
            "majority_threshold": selection.majority_threshold,
            "inference_stride": selection.inference_stride,
            "inner_summary": selection.inner_summary,
        }
    fused = fuse_scale_probabilities(frame_probabilities, fusion_weights)
    winner_indices = fused.argmax(axis=1)
    from phase_d_evaluation import FrameProbabilityResult

    result = FrameProbabilityResult(
        probabilities=fused,
        coverage=np.ones(len(fused), dtype=np.int64),
        labels=np.asarray(CLASSES, dtype=object)[winner_indices],
        confidence=fused[np.arange(len(fused)), winner_indices],
        classes=tuple(CLASSES),
    )
    return fitted_by_scale, result, {
        "outer_held_out_subject": outer_held_out_subject,
        "outer_train_subjects": list(train_subjects),
        "scale_selections": scale_audit,
        "fusion_weights": list(map(float, fusion_weights)),
    }
