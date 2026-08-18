from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from phase_d_classical import ClassicalConfig, TrainingOnlyFeatureSelector, fit_classical_model
from phase_d_evaluation import aggregate_window_probabilities
from tsfel_histgb_pipeline import CLASSES, WindowFeatures, concatenate_window_features, evaluate_predictions


PAIR_SEARCH_VERSION = "phase-d-pair-search-v1"


@dataclass(frozen=True)
class OuterConfigSelection:
    config: ClassicalConfig
    inner_metrics: pd.DataFrame
    audit: dict


def _parse_train_subjects(value: object) -> tuple[int, ...]:
    return tuple(sorted(int(part) for part in str(value).split("_") if part))


def select_outer_configs_from_pair_scores(
    scores: pd.DataFrame,
    subject_ids: Sequence[int],
    candidates: Sequence[ClassicalConfig],
) -> tuple[dict[int, OuterConfigSelection], dict]:
    subjects = tuple(sorted(int(subject) for subject in subject_ids))
    candidate_list = list(candidates)
    required_columns = {
        "train_subjects",
        "candidate_index",
        "validation_subject",
        "accuracy",
        "macro_f1",
        "abnormal_f1",
    }
    missing = sorted(required_columns - set(scores.columns))
    if missing:
        raise ValueError(f"Pair scores are missing columns: {missing}")
    if len(subjects) != 4 or len(set(subjects)) != 4:
        raise ValueError("Outer selection requires four unique development subjects")
    if not candidate_list:
        raise ValueError("At least one candidate is required")

    normalized = scores.copy()
    normalized["train_tuple"] = normalized["train_subjects"].map(_parse_train_subjects)
    normalized["candidate_index"] = normalized["candidate_index"].astype(int)
    normalized["validation_subject"] = normalized["validation_subject"].astype(int)
    selections: dict[int, OuterConfigSelection] = {}

    for outer_held_out in subjects:
        outer_train = tuple(subject for subject in subjects if subject != outer_held_out)
        inner_frames: list[pd.DataFrame] = []
        inner_splits: list[dict] = []
        for validation_subject in outer_train:
            train_subjects = tuple(subject for subject in outer_train if subject != validation_subject)
            fold_scores = normalized[
                (normalized["train_tuple"] == train_subjects)
                & (normalized["validation_subject"] == validation_subject)
            ].copy()
            if len(fold_scores) != len(candidate_list):
                raise ValueError(
                    f"Expected {len(candidate_list)} scores for train={train_subjects}, "
                    f"validation={validation_subject}; found {len(fold_scores)}"
                )
            if set(fold_scores["candidate_index"]) != set(range(len(candidate_list))):
                raise ValueError("Pair scores have missing or unknown candidate indices")
            inner_frames.append(fold_scores.drop(columns="train_tuple"))
            inner_splits.append(
                {"train_subjects": list(train_subjects), "validation_subject": validation_subject}
            )

        inner_metrics = pd.concat(inner_frames, ignore_index=True)
        summary = inner_metrics.groupby("candidate_index", as_index=False).agg(
            mean_accuracy=("accuracy", "mean"),
            worst_accuracy=("accuracy", "min"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_abnormal_f1=("abnormal_f1", "mean"),
        )
        best_index = int(
            summary.sort_values(
                ["mean_accuracy", "worst_accuracy", "mean_macro_f1", "mean_abnormal_f1"],
                ascending=False,
                kind="mergesort",
            ).iloc[0]["candidate_index"]
        )
        audit = {
            "outer_held_out_subject": outer_held_out,
            "outer_train_subjects": list(outer_train),
            "inner_splits": inner_splits,
            "selected_candidate_index": best_index,
            "selection_source": "resumable_pair_search",
        }
        selections[outer_held_out] = OuterConfigSelection(
            config=candidate_list[best_index],
            inner_metrics=inner_metrics,
            audit=audit,
        )

    leakage_ok = all(
        held_out not in split["train_subjects"] and split["validation_subject"] != held_out
        for held_out, selection in selections.items()
        for split in selection.audit["inner_splits"]
    )
    if not leakage_ok:
        raise ValueError("Outer held-out subject leaked into pair-score selection")
    return selections, {
        "outer_fold_count": len(selections),
        "candidate_count": len(candidate_list),
        "leakage_check_passed": leakage_ok,
    }


def _training_signature(
    train_subjects: tuple[int, ...],
    windows: WindowFeatures,
    max_budget: int,
) -> dict:
    columns_payload = json.dumps(list(windows.x.columns), ensure_ascii=True).encode("utf-8")
    return {
        "version": PAIR_SEARCH_VERSION,
        "train_subjects": list(train_subjects),
        "rows": len(windows.x),
        "columns": len(windows.x.columns),
        "columns_sha256": hashlib.sha256(columns_payload).hexdigest(),
        "label_counts": {str(label): int(count) for label, count in windows.y.value_counts().sort_index().items()},
        "max_budget": int(max_budget),
    }


def _selector_path(output_dir: Path, train_subjects: tuple[int, ...]) -> Path:
    code = "_".join(map(str, train_subjects))
    return output_dir / f"selector_pair_{code}.joblib"


def _candidate_path(output_dir: Path, train_subjects: tuple[int, ...], candidate_index: int) -> Path:
    code = "_".join(map(str, train_subjects))
    return output_dir / f"model_pair_{code}_candidate_{candidate_index:02d}.joblib"


def run_resumable_pair_search(
    subject_windows: dict[int, WindowFeatures],
    frame_truth: dict[int, Sequence[str]],
    candidates: Sequence[ClassicalConfig],
    output_dir: Path,
    random_state: int = 42,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    subject_ids = tuple(sorted(subject_windows))
    if len(subject_ids) != 4:
        raise ValueError("Pair search requires exactly four development subjects")
    if set(subject_ids) != set(frame_truth):
        raise ValueError("Window and frame-truth subject sets must match")
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one candidate is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    max_budget = max(candidate.feature_budget for candidate in candidate_list)
    audit = {
        "version": PAIR_SEARCH_VERSION,
        "selector_fits": 0,
        "model_fits": 0,
        "cache_hits": 0,
        "completed_pair_candidates": 0,
        "random_state": random_state,
    }
    all_scores: list[pd.DataFrame] = []

    for train_subjects in combinations(subject_ids, 2):
        train = concatenate_window_features([subject_windows[subject] for subject in train_subjects])
        signature = _training_signature(train_subjects, train, max_budget)
        selector_file = _selector_path(output_dir, train_subjects)
        selector = None
        if selector_file.exists() and not force:
            payload = joblib.load(selector_file)
            if payload.get("signature") == signature:
                selector = payload.get("selector")
        if selector is None:
            selector = TrainingOnlyFeatureSelector(
                feature_budget=max_budget,
                random_state=random_state + sum(train_subjects),
            ).fit(train.x, train.y)
            joblib.dump({"signature": signature, "selector": selector}, selector_file, compress=3)
            audit["selector_fits"] += 1

        validation_subjects = [subject for subject in subject_ids if subject not in train_subjects]
        for candidate_index, config in enumerate(candidate_list):
            candidate_file = _candidate_path(output_dir, train_subjects, candidate_index)
            candidate_signature = {
                "training": signature,
                "candidate_index": candidate_index,
                "config": config.to_dict(),
                "random_state": random_state,
            }
            cached_scores = None
            if candidate_file.exists() and not force:
                payload = joblib.load(candidate_file)
                if payload.get("signature") == candidate_signature:
                    cached_scores = payload.get("scores")
            if cached_scores is not None:
                all_scores.append(cached_scores)
                audit["cache_hits"] += 1
                audit["completed_pair_candidates"] += 1
                continue

            restricted = selector.restrict_budget(config.feature_budget)
            fitted = fit_classical_model(
                train,
                config=config,
                random_state=random_state + candidate_index + sum(train_subjects) * 100,
                prefit_selector=restricted,
            )
            rows: list[dict] = []
            for validation_subject in validation_subjects:
                validation = subject_windows[validation_subject]
                probabilities = fitted.predict_proba(validation.x, CLASSES)
                frame_result = aggregate_window_probabilities(
                    validation.meta,
                    probabilities,
                    n_frames=len(frame_truth[validation_subject]),
                    classes=CLASSES,
                )
                truth = np.asarray(frame_truth[validation_subject], dtype=object)
                valid = np.asarray([label in set(CLASSES) for label in truth], dtype=bool)
                metrics = evaluate_predictions(truth[valid], frame_result.labels[valid])
                rows.append(
                    {
                        "train_subjects": "_".join(map(str, train_subjects)),
                        "candidate_index": candidate_index,
                        "validation_subject": validation_subject,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "abnormal_f1": metrics["abnormal_f1"],
                    }
                )
            scores = pd.DataFrame(rows)
            joblib.dump(
                {"signature": candidate_signature, "model": fitted, "scores": scores},
                candidate_file,
                compress=3,
            )
            all_scores.append(scores)
            audit["model_fits"] += 1
            audit["completed_pair_candidates"] += 1

    combined = pd.concat(all_scores, ignore_index=True).sort_values(
        ["train_subjects", "candidate_index", "validation_subject"],
        kind="mergesort",
    ).reset_index(drop=True)
    combined.to_csv(output_dir / "pair_search_scores.csv", index=False)
    (output_dir / "pair_search_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return combined, audit
